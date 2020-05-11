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

try:
    file = pd.read_csv('dataset.csv')
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
file['char_count'] = file['lyrics'].apply(len)

# number of words per song
file['word_count'] = file['lyrics'].apply(word_counter)

# number of sentences per song
file['sentence_count'] = file['lyrics'].apply(sentence_counter)

# number of verses per song
file['verse_count'] = file.lyrics.str.count('\n') + 1

# avg number of characters
avg_chars = file['char_count'].mean()

# avg number of words
avg_words = file['word_count'].mean()

# avg number of sentences
avg_sentences = file['sentence_count'].mean()

# avg number of verses
avg_verses = file['verse_count'].mean()


genres = file['genre'].value_counts()
genres = pd.DataFrame(genres)
genres = genres.rename(columns={'genre': 'num_songs'})
##IMPORTANT##
tmp = file[['genre', 'char_count', 'verse_count',
            'word_count', 'sentence_count']].groupby('genre').mean()

genres = genres.join(tmp)
print(genres)
# avg number of chars per genre
#file.loc['custom_index', 'column_name']


#plt.bar(range(len(genres)), list(genres.values()), align='center')
#plt.xticks(range(len(genres)), list(genres.keys()))
# plt.show()
