import pandas as pd
import re
import matplotlib.pyplot as plt
import langdetect

from dataset_metadata import DatasetMetadata

df = pd.read_csv('lyrics.csv')
print('Initial number of songs: {}'.format(df.shape[0]))


## REMOVE YEAR COLUMN ##
df = df.drop('year', 1)
df = df.drop('artist', 1)
df = df.drop('song', 1)


## REMOVE SONGS WITH NO LYRICS ##
print('Removing songs with no lyrics')
df = df.dropna()
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE SONG WITH GENRE LISTED AS 'OTHER' OR 'NOT AVAILABLE' ##
df = df[df.genre != 'Other']
df = df[df.genre != 'Not Available']


## REMOVE SONG WITH ONLY ONE VERSE ##
print('Removing songs with only one verse')
df = df[df.lyrics.str.contains('\n')]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE LYRICS WITH NO SPACES INBETWEEN WORDS ##
print('Removing songs with no spaces inbetween words')
df = df[df.lyrics.str.count(' ') != 0]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE SONGS NOT IN ENGLISH ##
print('Removing non English songs')


def lang_detector(song):
    try:
        language = langdetect.detect(" ".join(song.split('\n')[:3]))
    except:
        return False
    if(language == 'en'):
        return True
    else:
        return False


df['isEnglish'] = df['lyrics'].apply(lang_detector)
df = df[df['isEnglish'] == True]
df = df.drop('isEnglish', 1)
print('Current number of songs: {}'.format(df.shape[0]))


## RE-REMOVING ONE VERSE SONGS AND EMPTY SONGS ##
print('re-removing one verse songs and empty songs')
df = df.dropna()
df = df[df.lyrics.str.contains('\n')]
print('Current number of songs: {}'.format(df.shape[0]))
df['char_count'] = df['lyrics'].apply(len)
print('current char mean: {}'.format(df['char_count'].mean()))


## REMOVE VERSES OF ONLY ONE WORD ##
print('Removing verses of only one word')


def removeOneWordVerses(song):
    line_list = song.split('\n')
    for line in line_list:
        if(len(line.split(' ')) == 1):
            line_list.remove(line)
    return '\n'.join(line_list)


df['lyrics'] = df['lyrics'].apply(removeOneWordVerses)
df['char_count'] = df['lyrics'].apply(len)
print('current char mean: {}'.format(df['char_count'].mean()))


## REMOVE LINES WITH META WORDS ##
print('Removing meta words')


def removeMetaWords(song):
    line_list = song.split('\n')
    for line in line_list:
        r = re.compile(r'chorus|bridge|outro|intro|verse|\[|\{|\(', flags=re.I)
        if(r.findall(line) and len(line) < 12):
            line_list.remove(line)
    return '\n'.join(line_list)


df['lyrics'] = df['lyrics'].apply(removeMetaWords)
df['char_count'] = df['lyrics'].apply(len)
print('current char mean: {}'.format(df['char_count'].mean()))


## REMOVE NON ASCII CHARACTERS ##
print('Removing non ASCII characters')


def removeNonASCII(song):
    return ''.join([i if ord(i) < 128 else '' for i in song])


df['lyrics'] = df['lyrics'].apply(removeNonASCII)
df['char_count'] = df['lyrics'].apply(len)
print('current char mean: {}'.format(df['char_count'].mean()))


print('re-removing one verse songs and empty songs')
df = df.dropna()
df = df[df.lyrics.str.contains('\n')]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE SONGS WITH TOO MANY OR NOT ENOUGH VERSES ##
df['line_count'] = df.lyrics.str.count('\n') + 1
tmp_mean = df[['genre', 'line_count']].groupby('genre').mean()
tmp_std = df[['genre', 'line_count']].groupby('genre').std()
print('Removing songs with too many or not enough verses')


def getMean(genre):
    return tmp_mean.loc[genre]['line_count']


def getStd(genre):
    return tmp_std.loc[genre]['line_count']


df['mean'] = df['genre'].apply(getMean)
df['std'] = df['genre'].apply(getStd)

df['upper_limit'] = df['mean'] + df['std']
df['lower_limit'] = df['mean'] - df['std']

df = df[df['line_count'] < df['upper_limit']]
df = df[df['line_count'] > df['lower_limit']]
print('Current number of songs: {}'.format(df.shape[0]))


## DROP IRRELEVANT COLUMNS ##
df = df.drop('line_count', 1)
df = df.drop('char_count', 1)
df = df.drop('mean', 1)
df = df.drop('std', 1)
df = df.drop('upper_limit', 1)
df = df.drop('lower_limit', 1)


## CHANGE GENRE COLUMN FROM STRING TO NUMBER ##


## SAVE FILE TO CSV ##
df.to_csv('dataset.csv', index=False)
print('Final number of songs: {}'.format(df.shape[0]))
print('########### CLEANING COMPLETE ###########')


# ## REMOVE SONGS WITH TOO MANY OR NOT ENOUGH VERSES ##
# print('Removing songs with too many or too few verses')
# df['verse_count'] = df.lyrics.str.count('\n') + 1
# avg_verses = df['verse_count'].mean()
# df = df[df.lyrics.str.count('\n') > avg_verses * 0.05]
# df = df[df.lyrics.str.count('\n') < avg_verses * 1.95]
# print('Current number of songs: {}'.format(df.shape[0]))


# ## REMOVE SHORT AND LONG LYRICS ##
# print('Removing songs with very short or very longs lyrics')
# df['char_count'] = df['lyrics'].apply(len)
# avg_chars = df['char_count'].mean()
# df = df[df.char_count > avg_chars * 0.05]
# df = df[df.char_count < avg_chars * 1.95]
# print('Current number of songs: {}'.format(df.shape[0]))
