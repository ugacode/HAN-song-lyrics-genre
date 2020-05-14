# TODO
# average number of chars per verse per genre
# average number of chars per word per genre
# average number of words per sentence per genre
# average number of words per verse per genre
# average number of words per genre
# average number of sentences per genre
# average number of verses per genre

import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

from learning_dataset_generator import LEARNING_DATASET_TRAIN_PATH, LEARNING_DATASET_TEST_PATH, \
    LEARNING_SMALL_DATASET_TRAIN_PATH

try:
    df = pd.read_csv(LEARNING_DATASET_TRAIN_PATH)
except:
    print('Dataset.csv not found')
    sys.exit()


def word_counter(song):
    sentence_list = re.split('\n|\s', song)
    return len(list(filter(None, sentence_list)))


def sentence_counter(song):
    # probably will also add , as a separator of sentences
    sentence_list = re.split('[?!.:;]\n', song)
    return len(list(filter(None, sentence_list)))


# number of characters per song
df['char_count'] = df['lyrics'].apply(len)

# number of words per song
df['word_count'] = df['lyrics'].apply(word_counter)

# number of sentences per song
df['sentence_count'] = df['lyrics'].apply(sentence_counter)

# number of verses per song
df['verse_count'] = df.lyrics.str.count('\n') + 1

# avg number of characters
avg_chars = df['char_count'].mean()

# avg number of words
avg_words = df['word_count'].mean()

# avg number of sentences
avg_sentences = df['sentence_count'].mean()

# avg number of verses
avg_verses = df['verse_count'].mean()


genres = df['genre'].value_counts()
genres = pd.DataFrame(genres)
genres = genres.rename(columns={'genre': 'num_songs'})
##IMPORTANT##
tmp = df[['genre', 'char_count', 'verse_count',
            'word_count', 'sentence_count']].groupby('genre').mean()

genres = genres.join(tmp)
print(genres)
# avg number of chars per genre
#df.loc['custom_index', 'column_name']


#plt.bar(range(len(genres)), list(genres.values()), align='center')
#plt.xticks(range(len(genres)), list(genres.keys()))
# plt.show()
