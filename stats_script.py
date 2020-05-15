import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

try:
    df = pd.read_csv('dataset.csv')
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
# avg number of chars per genre
#df.loc['custom_index', 'column_name']


# plt.bar(range(len(genres)), list(genres.values()), align='center')
# plt.xticks(range(len(genres)), list(genres.keys()))
# plt.show()
