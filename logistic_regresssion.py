import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


WORD_EMBEDDING_SIZE = 100
LEARNING_RATE = 0.001  # experiment with this
BATCH_SIZE = 128


def test_network(model, test_dataset):
    with torch.no_grad():
        data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        correct_predictions = 0
        for _, sample_batched in enumerate(data_loader):
            probability = model(sample_batched['lyrics'])
            _, prediction = torch.max(probability, 1)
            if (prediction.item() == sample_batched['genre']):
                correct_predictions += 1
        accuracy = (correct_predictions / len(data_loader)) * 100
        return accuracy


def train_network(model, training_dataset, epochs):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        model.zero_grad()
        for _, sample_batched in enumerate(data_loader):
            training_batch = sample_batched['lyrics']
            label_batch = sample_batched['genre']
            probabilities = model(training_batch)
            loss = loss_function(probabilities, label_batch)
            loss.backward()
            optimizer.step()

    return model


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, genre_count=10):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(WORD_EMBEDDING_SIZE, genre_count)

    def forward(self, song_word_average_batch):
        return F.softmax(self.linear(song_word_average_batch), dim=1)
