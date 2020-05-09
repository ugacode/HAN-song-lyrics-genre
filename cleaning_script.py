import csv
import math
import pandas as pd

file = pd.read_csv('lyrics.csv')
print('Initial number of rows: {}'.format(file.shape[0]))


## REMOVE YEAR COLUMN ##
file = file.drop('year', 1)


## REMOVE SONGS WITH NO LYRICS ##
print('Removing songs with no lyrics')
file = file.dropna()
print('Current rows: {}'.format(file.shape[0]))


## REMOVE SONG WITH ONLY ONE VERSE ##
print('Removing songs with only one verse')
file = file[file.lyrics.str.contains('\n')]
print('Current rows: {}'.format(file.shape[0]))


## REMOVE SONGS WITH TOO MANY OR NOT ENOUGH VERSES ##
print('Removing songs with too many or too few verses')
file['verse_count'] = file.lyrics.str.count('\n') + 1
av_verses = file['verse_count'].mean()
file = file[file.lyrics.str.count('\n') > av_verses * 0.05]
file = file[file.lyrics.str.count('\n') < av_verses * 1.95]
print('Curent rows: {}'.format(file.shape[0]))


## REMOVE SHORT AND LONG LYRICS ##
print('Removing songs with very short or very longs lyrics')
file['lyrics_length'] = file['lyrics'].apply(len)
av_chars = file['lyrics_length'].mean()
file = file[file.lyrics_length > av_chars * 0.05]
file = file[file.lyrics_length < av_chars * 1.95]
print('Current rows: {}'.format(file.shape[0]))


## REMOVE LYRICS WITH NO SPACES INBETWEEN WORDS ##
print('Removing songs with no spaces inbetween words')
file = file[file.lyrics.str.count(' ') != 0]
print('Current rows: {}'.format(file.shape[0]))

file['verse_count'] = file.lyrics.str.count('\n') + 1
file['lyrics_length'] = file['lyrics'].apply(len)

# todo
# english only
# total number of words per song
# total number of sentences per song
# average number of words
# average number of sentences
# average length of verse per song
# average number of verses
# total number of verses per song

# average number of words per genre
# average number of sentences per genre
# average number of verses per genre

# amount of genres
# amount of subgenres

# print('Dataset stats: ')
file['word_count'] = len(file.lyrics.split())
av_words = file['word_count'].mean()


def sentence_counter(song):
    sentences = re.split(', |. |\n', song)
    return sum(1 for sentence in sentences if sentence)


# fix this
file['sentence_counter'] = file['lyrics'].apply(sentence_counter)


print(file['sentence_counter'])

# print()

print('########### CLEANING COMPLETE ###########')
file.to_csv('cleaned.csv', index=False)
