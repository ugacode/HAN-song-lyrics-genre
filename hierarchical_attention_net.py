import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, ConfusionMatrix

HIDDEN_SIZE = 50
NUM_CLASSES = 10
EPOCHS = 10
WORD_EMBEDDING_SIZE = 100
LEARNING_RATE = 0.001  # experiment with this
BATCH_SIZE = 128


def test_network(model, test_dataset):
    with torch.no_grad():
        model.eval()
        correct_predictions = 0
        data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for lyrics, genres in data_loader:
            samples_count = len(genres)
            model._init_hidden_state(samples_count)
            _, predict = torch.max(model(lyrics), dim=1)

            for i in range(samples_count):
                if (predict[i] == genres[i]):
                    correct_predictions += 1

        return (correct_predictions / len(test_dataset)) * 100


def train_network(model, training_dataset, epochs):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        model.zero_grad()
        for lyrics, genres in data_loader:
            samples_count = len(genres)
            model._init_hidden_state(samples_count)
            probabilities = model(lyrics)
            loss = loss_function(probabilities, genres)
            loss.backward()
            optimizer.step()

    return model


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


class WordAttEncoder(nn.Module):
    def __init__(self, word_embedding, hidden_size):
        super(WordAttEncoder, self).__init__()
        vocab_size, embed_size = word_embedding.shape
        vocab_size += 1
        unknown_word_embedding = np.zeros((1, embed_size))
        word_embedding = torch.from_numpy(np.concatenate([unknown_word_embedding, word_embedding], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size).from_pretrained(
            word_embedding, freeze=True)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.embedding(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


class SentAttEncoder(nn.Module):
    def __init__(self, sent_hidden_size, word_hidden_size, num_classes):
        super(SentAttEncoder, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output


class HAN(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word_embedding):
        super(HAN, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttEncoder(pretrained_word_embedding, word_hidden_size)
        self.sent_att_net = SentAttEncoder(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        return output
