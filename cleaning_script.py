import pandas as pd
import re
import matplotlib.pyplot as plt
import langdetect

pd = pd.read_csv('lyrics.csv')
print('Initial number of songs: {}'.format(pd.shape[0]))


## REMOVE YEAR COLUMN ##
pd = pd.drop('year', 1)


## REMOVE SONGS WITH NO LYRICS ##
print('Removing songs with no lyrics')
pd = pd.dropna()
print('Current number of songs: {}'.format(pd.shape[0]))


## REMOVE SONG WITH GENRE LISTED AS 'OTHER' OR 'NOT AVAILABLE' ##
pd = pd[pd.genre != 'Other']
pd = pd[pd.genre != 'Not Available']


## REMOVE SONG WITH ONLY ONE VERSE ##
print('Removing songs with only one verse')
pd = pd[pd.lyrics.str.contains('\n')]
print('Current number of songs: {}'.format(pd.shape[0]))


## REMOVE SONGS WITH TOO MANY OR NOT ENOUGH VERSES ##
print('Removing songs with too many or too few verses')
pd['verse_count'] = pd.lyrics.str.count('\n') + 1
avg_verses = pd['verse_count'].mean()
pd = pd[pd.lyrics.str.count('\n') > avg_verses * 0.05]
pd = pd[pd.lyrics.str.count('\n') < avg_verses * 1.95]
print('Current number of songs: {}'.format(pd.shape[0]))


## REMOVE SHORT AND LONG LYRICS ##
print('Removing songs with very short or very longs lyrics')
pd['char_count'] = pd['lyrics'].apply(len)
avg_chars = pd['char_count'].mean()
pd = pd[pd.char_count > avg_chars * 0.05]
pd = pd[pd.char_count < avg_chars * 1.95]
print('Current number of songs: {}'.format(pd.shape[0]))


## REMOVE LYRICS WITH NO SPACES INBETWEEN WORDS ##
print('Removing songs with no spaces inbetween words')
pd = pd[pd.lyrics.str.count(' ') != 0]
print('Current number of songs: {}'.format(pd.shape[0]))


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


pd['isEnglish'] = pd['lyrics'].apply(lang_detector)
pd = pd[pd['isEnglish'] == True]
print('Final number of songs: {}'.format(pd.shape[0]))


## DROP OUTDATED COLUMNS ##
pd = pd.drop('verse_count', 1)
pd = pd.drop('char_count', 1)
pd = pd.drop('isEnglish', 1)


## SAVE FILE TO CSV ##
pd.to_csv('dataset.csv', index=False)
print('########### CLEANING COMPLETE ###########')
