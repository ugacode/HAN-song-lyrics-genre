import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

WORD_EMBEDDING_SIZE = 100
LEARNING_RATE = 0.01    # experiment with this
BATCH_SIZE = 128


def test_network(model, test_data, test_labels):
    with torch.no_grad():
        correct_predictions = 0
        test_data_size = len(test_data)
        for i in range(test_data_size):
            lyrics = test_data[i]
            label = test_labels[i]
            word_vec = LogisticRegressionClassifier.song_lyrics_to_word_average(lyrics)
            probability = model(word_vec)
            _, prediction = torch.max(probability, 1)
            if (prediction == label):
                correct_predictions += 1
        accuracy = correct_predictions / test_data_size
        print(f'accuracy - {accuracy}')
        return accuracy


def train_network(train_data, train_labels, epochs):
    model = LogisticRegressionClassifier()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.zero_grad()

        batch_count = int(len(train_data) / BATCH_SIZE)
        for batch_id in range(batch_count):
            start_index = batch_id * BATCH_SIZE
            end_index = (batch_id + 1) * BATCH_SIZE
            train_batch = train_data[:, start_index : end_index]
            hot_batch = train_labels[:, start_index:end_index]

            probabilities = model(train_batch)
            loss = loss_function(probabilities, hot_batch)
            loss.backward()
            optimizer.step()

    return model


def word_to_glove(word):
    return np.random.randn(WORD_EMBEDDING_SIZE).reshape(1, -1)


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, genre_count=10):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.linear(WORD_EMBEDDING_SIZE, genre_count)

    def forward(self, song_word_average_batch):
        return F.softmax(self.linear(song_word_average_batch), dim=1)

    @staticmethod
    def song_lyrics_to_word_average(song_lyrics):
        return torch.from_numpy(np.mean(np.array([word_to_glove(word) for word in song_lyrics]), axis=0))


