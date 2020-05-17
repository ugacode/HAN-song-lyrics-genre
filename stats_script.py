import re
import statistics
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk import word_tokenize
from collections import Counter
from learning_dataset_generator import LEARNING_DATASET_TRAIN_PATH, LEARNING_DATASET_TEST_PATH, \
    LEARNING_SMALL_DATASET_TRAIN_PATH, LEARNING_TINY_DATASET_TRAIN_PATH, LEARNING_TINY_DATASET_TEST_PATH, \
    LEARNING_SMALL_DATASET_TEST_PATH

try:
    df = pd.read_csv(FULL_DATASET_PATH)
except:
    print('Dataset.csv not found')
    sys.exit()


def word_counter(song):
    line_list = re.split('\n|\s', song)
    return len(list(filter(None, line_list)))


# number of characters per song
df['char_count'] = df['lyrics'].apply(len)

# number of words per song
df['word_count'] = df['lyrics'].apply(word_counter)

# number of verses per song
df['line_count'] = df.lyrics.str.count('\n') + 1

# avg number of characters
avg_chars = df['char_count'].mean()

# avg number of words
avg_words = df['word_count'].mean()


# avg number of line
avg_lines = df['line_count'].mean()


genres = df['genre'].value_counts()
genres = pd.DataFrame(genres)
genres = genres.rename(columns={'genre': 'num_songs'})
##IMPORTANT##
tmp = df[['genre', 'char_count', 'word_count',
          'line_count']].groupby('genre').mean()

tmp = tmp.rename(columns={'char_count': 'char_avg',
                          'word_count': 'word_avg', 'line_count': 'line_avg'})

genres = genres.join(tmp)
print(genres)
print('Dataset shape {}'.format(df.shape))

word_series = pd.Series()
word_array = []


def splitter(song):
    line_list = song.split('\n')
    output = []
    for line in line_list:
        output.append(len(word_tokenize(line)))
    return np.array(output)


word_array_series = df['lyrics'].apply(splitter)

word_array = np.array([0])
for array in word_array_series:
    word_array = np.concatenate((word_array, array))

word_array_series = pd.Series(word_array)
word_count_series = word_array_series.value_counts()
print('the most common number of words is: {}'.format(
    word_count_series[word_count_series == max(word_count_series)]))

word_count_series = word_count_series.sort_index()
word_count_series.plot.bar()
plt.show()


line_count_series = df['line_count'].value_counts()
print('the most common number of lines is: {}'.format(
    line_count_series[line_count_series == max(line_count_series)]))
line_count_series = line_count_series.sort_index()
line_count_series.plot.bar()
plt.show()
