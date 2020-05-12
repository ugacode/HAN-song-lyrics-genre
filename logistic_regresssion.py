import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from nltk import sent_tokenize, word_tokenize

from word_encoding import WordEncodingAuto

WORD_EMBEDDING_SIZE = 100
LEARNING_RATE = 0.01  # experiment with this
BATCH_SIZE = 32


def lyrics_to_words(lyrics):
    lines = lyrics.replace('\n', ' ')
    words = word_tokenize(lines)
    return words


def test_network(model, test_data, test_labels):
    with torch.no_grad():
        correct_predictions = 0
        test_data_size = len(test_data)
        for i in range(test_data_size):
            lyrics = test_data[i]
            label = test_labels[i]
            word_vec = model.song_lyrics_to_word_average(lyrics).reshape(1, -1)
            probability = model(word_vec)
            _, prediction = torch.max(probability, 1)
            if (prediction.item() == label.item()):
                correct_predictions += 1
        accuracy = (correct_predictions / test_data_size) * 100
        return accuracy


def train_network(model, train_data, train_labels, epochs):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.zero_grad()

        batch_count = int(len(train_data) / BATCH_SIZE)
        for batch_id in range(batch_count):
            start_index = batch_id * BATCH_SIZE
            end_index = (batch_id + 1) * BATCH_SIZE
            if (end_index > len(train_data)):
                end_index = len(train_data)
            train_batch = train_data[start_index: end_index, :]
            label_batch = torch.from_numpy(train_labels[start_index:end_index])

            probabilities = model(train_batch)
            loss = loss_function(probabilities, label_batch)
            loss.backward()
            optimizer.step()

    return model


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, genre_count=10):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(WORD_EMBEDDING_SIZE, genre_count)
        self.word_encoder = WordEncodingAuto()

    def word_to_glove(self, word):
        return self.word_encoder.get_word_vector(word).reshape(1, -1)

    def forward(self, song_word_average_batch):
        return F.softmax(self.linear(song_word_average_batch), dim=1)

    def song_lyrics_to_word_average(self, song_lyrics):
        return torch.from_numpy(np.mean(np.array(torch.cat([self.word_to_glove(word) for word in song_lyrics]).T), axis=1))
