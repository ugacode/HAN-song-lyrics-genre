import pandas as pd
import re
import matplotlib.pyplot as plt
import langdetect

df = pd.read_csv('lyrics.csv')
print('Initial number of songs: {}'.format(df.shape[0]))


## REMOVE YEAR COLUMN ##
df = df.drop('year', 1)


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


## REMOVE SONGS WITH TOO MANY OR NOT ENOUGH VERSES ##
print('Removing songs with too many or too few verses')
df['verse_count'] = df.lyrics.str.count('\n') + 1
avg_verses = df['verse_count'].mean()
df = df[df.lyrics.str.count('\n') > avg_verses * 0.05]
df = df[df.lyrics.str.count('\n') < avg_verses * 1.95]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE SHORT AND LONG LYRICS ##
print('Removing songs with very short or very longs lyrics')
df['char_count'] = df['lyrics'].apply(len)
avg_chars = df['char_count'].mean()
df = df[df.char_count > avg_chars * 0.05]
df = df[df.char_count < avg_chars * 1.95]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE LYRICS WITH NO SPACES INBETWEEN WORDS ##
print('Removing songs with no spaces inbetween words')
df = df[df.lyrics.str.count(' ') != 0]
print('Current number of songs: {}'.format(df.shape[0]))


## REMOVE SONGS NOT IN ENGLISH ##
print('Removing non English songs')


def lang_detector(song):
    try:
        language = langdetect.detect(song[:len(song) // 5])
    except:
        return False
    if(language == 'en'):
        return True
    else:
        return False


df['isEnglish'] = df['lyrics'].apply(lang_detector)
df = df[df['isEnglish'] == True]
print('Final number of songs: {}'.format(df.shape[0]))


## DROP OUTDATED COLUMNS ##
df = df.drop('verse_count', 1)
df = df.drop('char_count', 1)
df = df.drop('isEnglish', 1)


## SAVE FILE TO CSV ##
df.to_csv('dataset.csv', index=False)
print('########### CLEANING COMPLETE ###########')
