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
    # probably will also add , as a separator of sentences
    sentence_list = re.split('\n|\s', song)
    return len(list(filter(None, sentence_list)))


def sentence_counter(song):
    # probably will also add , as a separator of sentences
    sentence_list = re.split('[?!.:;]\n', song)
    return len(list(filter(None, sentence_list)))

def genre_counter(genre):
    if genre not in genres:
        genres[genre] = 1
    else:
        genres[genre] += 1


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

genres = {}


print('almost done')
file['genre'].apply(genre_counter)





plt.bar(range(len(genres)), list(genres.values()), align='center')
plt.xticks(range(len(genres)), list(genres.keys()))
plt.show()
