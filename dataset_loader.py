import torch
from torch.utils.data import Dataset
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import numpy as np

from word_encoding import WordEncodingAuto

MAX_LINES = 60
MAX_WORDS = 10

WORD_ENCODE = WordEncodingAuto()



def lyrics_to_words_lines(lyrics):
    lines = lyrics.split('\n')
    words = [list(map(str.lower, word_tokenize(line))) for line in lines]
    lines_count = len(words)
    if (lines_count < MAX_LINES):
        for _ in range(lines_count, MAX_LINES):
            words.append(["PAD"])
    for line_i in range(MAX_LINES):
        current_line = words[line_i]
        line_length = len(current_line)
        if (line_length < MAX_WORDS):
            for _ in range(line_length, MAX_WORDS):
                words[line_i].append("PAD")
        for word_i in range(MAX_WORDS):
            words[line_i][word_i] = WORD_ENCODE.get_word_vector(words[line_i][word_i])
    return words


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


class LyricsDatasetHAN(Dataset):
    def __init__(self, csv_file, transform=None, transform_dynamically=False):
        self.word_encoder = WordEncodingAuto()
        data = pd.read_csv(csv_file)
        self.data_size = data.shape[0]
        self.lyrics = data['lyrics'].transform(lyrics_to_words_lines).to_numpy()
        self.genres = data['genre'].to_numpy()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.lyrics[idx], self.genres[idx])
        return sample


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

#lyr = "Oh baby, how you doing?\nYou know I'm gonna cut right to the chase\nSome women were made but me, myself\nI like to think that I was created for a special purpose\nYou know, what's more special than you? You feel me\nIt's on baby, let's get lost\nYou don't need to call into work 'cause you're the boss\nFor real, want you to show me how you feel\nI consider myself lucky, that's a big deal\nWhy? Well, you got the key to my heart\nBut you ain't gonna need it, I'd rather you open up my body\nAnd show me secrets, you didn't know was inside\nNo need for me to lie\nIt's too big, it's too wide\nIt's too strong, it won't fit\nIt's too much, it's too tough\nHe talk like this 'cause he can back it up\nHe got a big ego, such a huge ego\nI love his big ego, it's too much\nHe walk like this 'cause he can back it up\nUsually I'm humble, right now I don't choose\nYou can leave with me or you could have the blues\nSome call it arrogant, I call it confident\nYou decide when you find on what I'm working with\nDamn I know I'm killing you with them legs\nBetter yet them thighs\nMatter a fact it's my smile or maybe my eyes\nBoy you a site to see, kind of something like me\nIt's too big, it's too wide\nIt's too strong, it won't fit\nIt's too much, it's too tough\nI talk like this 'cause I can back it up\nI got a big ego, such a huge ego\nBut he love my big ego, it's too much\nI walk like this 'cause I can back it up\nI, I walk like this 'cause I can back it up\nI, I talk like this 'cause I can back it up\nI, I can back it up, I can back it up\nI walk like this 'cause I can back it up\nIt's too big, it's too wide\nIt's too strong, it won't fit\nIt's too much, it's too tough\nHe talk like this 'cause he can back it up\nHe got a big ego, such a huge ego, such a huge ego\nI love his big ego, it's too much\nHe walk like this 'cause he can back it up\nEgo so big, you must admit\nI got every reason to feel like I'm that bitch\nEgo so strong, if you ain't know\nI don't need no beat, I can sing it with piano"
#w = lyrics_to_words_lines(lyr)
#a = 42