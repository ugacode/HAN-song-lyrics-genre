import torch
from torch.utils.data import Dataset
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import numpy as np

from word_encoding import WordEncodingAuto


def lyrics_to_words(lyrics):
    words = word_tokenize(lyrics)
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

        sample =  (self.lyrics[idx], self.genres[idx])
        if (self.transform and (self.transform_dynamically == True)):
            sample = self.transform(sample)

        return sample
