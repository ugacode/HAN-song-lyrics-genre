import pandas as pd
import re
import matplotlib.pyplot as plt
import langdetect

file = pd.read_csv('lyrics.csv')
print('Initial number of rows: {}'.format(file.shape[0]))


## REMOVE YEAR COLUMN ##
file = file.drop('year', 1)


## REMOVE SONGS WITH NO LYRICS ##
print('Removing songs with no lyrics')
file = file.dropna()
print('Current rows: {}'.format(file.shape[0]))


## REMOVE SONG WITH GENRE LISTED AS 'OTHER' OR 'NOT AVAILABLE' ##
file = file[file.genre != 'Other']
file = file[file.genre != 'Not Available']


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


file['isEnglish'] = file['lyrics'].apply(lang_detector)
file = file[file['isEnglish'] == True]
print('Current rows: {}'.format(file.shape[0]))


##DROP OUTDATED COLUMNS ##
file = file.drop('verse_count', 1)
file = file.drop('char_count', 1)
file = file.drop('isEnglish', 1)


## SAVE FILE TO CSV ##
file.to_csv('dataset.csv', index=False)
print('########### CLEANING COMPLETE ###########')
