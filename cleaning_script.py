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
avg_verses = file['verse_count'].mean()
file = file[file.lyrics.str.count('\n') > avg_verses * 0.05]
file = file[file.lyrics.str.count('\n') < avg_verses * 1.95]
print('Curent rows: {}'.format(file.shape[0]))


## REMOVE SHORT AND LONG LYRICS ##
print('Removing songs with very short or very longs lyrics')
file['char_count'] = file['lyrics'].apply(len)
avg_chars = file['char_count'].mean()
file = file[file.char_count > avg_chars * 0.05]
file = file[file.char_count < avg_chars * 1.95]
print('Current rows: {}'.format(file.shape[0]))


## REMOVE LYRICS WITH NO SPACES INBETWEEN WORDS ##
print('Removing songs with no spaces inbetween words')
file = file[file.lyrics.str.count(' ') != 0]
print('Current rows: {}'.format(file.shape[0]))

# todo
# english only
# average length of verse per song
# average number of words per genre
# average number of sentences per genre
# average number of verses per genre
# amount of genres
# amount of subgenres

def word_counter(song):
    # probably will also add , as a separator of sentences
    sentence_list = re.split('\n|\s', song)
    return list(filter(None, sentence_list))


def sentence_counter(song):
    # probably will also add , as a separator of sentences
    sentence_list = re.split('[?!.:;]\n', song)
    return list(filter(None, sentence_list))


#number of characters per song
file['char_count'] = file['lyrics'].apply(len)

#number of words per song
file['word_count'] = file['lyrics'].apply(word_counter)

#number of sentences per song
file['sentence_count'] = file['lyrics'].apply(sentence_counter)

#number of verses per song
file['verse_count'] = file.lyrics.str.count('\n') + 1

#avg number of characters
avg_chars = file['char_count'].mean()

#avg number of words
avg_words = file['word_count'].mean()

#avg number of sentences
avg_sentences = file['sentence_count'].mean()

#avg number of verses
avg_verses = file['verse_count'].mean()






# print()

print('########### CLEANING COMPLETE ###########')
file.to_csv('cleaned.csv', index=False)
