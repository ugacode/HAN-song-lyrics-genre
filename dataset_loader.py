import torch
from torch.utils.data import Dataset
import pandas as pd
from nltk import word_tokenize
import numpy as np
import csv

from word_encoding import WordEncodingAuto

MAX_LINES = 40  # based on distribution in dataset
MAX_WORDS = 10  # based on distribution in dataset


def lyrics_to_words(lyrics):
    words = [word.lower() for word in word_tokenize(lyrics)]
    return words


class WordAverageTransform(object):
    word_encoder = WordEncodingAuto()

    def __call__(self, sample):
        lyrics = sample[0]
        word_average = torch.from_numpy(
            np.mean(np.array(torch.cat([self.word_to_glove(word) for word in lyrics]).T), axis=1))
        trans_sample = (word_average, sample[1])
        return trans_sample

    @staticmethod
    def word_to_glove(word):
        return WordAverageTransform.word_encoder.get_word_vector(word).reshape(1, -1)


class LyricsDataset(Dataset):
    def __init__(self, csv_file, transform=None, transform_dynamically=False):
        data = pd.read_csv(csv_file)
        self.data_size = data.shape[0]
        self.lyrics = data['lyrics'].transform(lyrics_to_words).to_numpy()
        self.genres = data['genre'].to_numpy()
        self.transform = transform
        self.transform_dynamically = transform_dynamically
        if (self.transform and (self.transform_dynamically == False)):
            for i in range(self.data_size):
                transformed = self.transform((self.lyrics[i], self.genres[i]))
                self.lyrics[i] = transformed[0]
                self.genres[i] = transformed[1]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.lyrics[idx], self.genres[idx])
        if (self.transform and (self.transform_dynamically == True)):
            sample = self.transform(sample)

        return sample


class LyricsDatasetEmbeddedHAN(Dataset):
    def __init__(self, data_csv_path, embedding_path):
        super(LyricsDatasetEmbeddedHAN, self).__init__()

        lyrics, genres = [], []
        data = pd.read_csv(data_csv_path)
        for _, row in data.iterrows():
            lyrics.append(row["lyrics"].lower())
            genres.append(row["genre"])

        self.lyrics = lyrics
        self.genres = genres
        self.vocab = pd.read_csv(filepath_or_buffer=embedding_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                 usecols=[0]).values
        self.vocab = [word[0] for word in self.vocab]
        self.data_size = len(self.genres)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        genre = self.genres[idx]
        lyrics = self.lyrics[idx]
        lyrics_encode = [
            [self.vocab.index(word) if word in self.vocab else -1 for word in word_tokenize(text=lines)] for lines
            in
            lyrics.split('\n')]

        # pad words
        for lines in lyrics_encode:
            if len(lines) < MAX_WORDS:
                word_padding = [-1 for _ in range(MAX_WORDS - len(lines))]
                lines.extend(word_padding)

        # pad lines
        if len(lyrics_encode) < MAX_LINES:
            line_padding = [[-1 for _ in range(MAX_WORDS)] for _ in range(MAX_LINES - len(lyrics_encode))]
            lyrics_encode.extend(line_padding)

        # truncate
        lyrics_encode = [lines[:MAX_WORDS] for lines in lyrics_encode][:MAX_LINES]

        lyrics_encode = np.stack(arrays=lyrics_encode, axis=0)
        lyrics_encode += 1

        return lyrics_encode.astype(np.int64), genre
