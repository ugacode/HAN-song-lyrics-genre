import datetime

from dataset_metadata import DatasetMetadata, JSON_FILE_PATH
from majority_classifier import MajorityClassifier
import logistic_regresssion

import torch

import pandas as pd


MOCK_DATASET_PATH = '..\\lyrics\\MOCK.csv'
LR_MODEL_PATH = '.\\models\\logistic_regression.model'


def write_dataset_metadata():
    dm = DatasetMetadata()
    dm.dump_to_file(JSON_FILE_PATH)


def load_dataset_from_file():
    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    print(dm.genre_labels)
    print(dm.song_count)
    print(dm.genre_labels[dm.most_common_genre_id])


def test_majority_classifier():
    classifier = MajorityClassifier(JSON_FILE_PATH)
    print(f'Majority classifier test = {classifier.forward("THESE ARE LYRICS")}')


def train_model_and_save(model_path, train_data, train_labels, test_data, test_labels):
    model = logistic_regresssion.LogisticRegressionClassifier()
    accuracy = logistic_regresssion.test_network(model, test_data, test_labels)
    print(f'{datetime.datetime.now()} - model accuracy before training - {accuracy}')
    word_average_train_data = [model.song_lyrics_to_word_average(song) for song in train_data]
    word_average_train_data_combined = torch.stack(word_average_train_data, dim=0)
    model = logistic_regresssion.train_network(model, word_average_train_data_combined, train_labels, 10)
    torch.save(model.state_dict(), model_path)


def load_model_and_test(model_path, test_data, test_labels):
    model = logistic_regresssion.LogisticRegressionClassifier()
    model.load_state_dict(torch.load(model_path))
    accuracy = logistic_regresssion.test_network(model, test_data, test_labels)
    return accuracy


def test_logistic_regression():
    print(f'{datetime.datetime.now()} - starting logistic regression testing')
    mock_data = pd.read_csv(MOCK_DATASET_PATH)
    data_size = mock_data.shape[0]
    train_index = int(data_size * 0.75)
    # split into labels and lyrics
    lyrics = mock_data['lyrics'].transform(logistic_regresssion.lyrics_to_words)
    labels = mock_data['genre']
    lyrics_train = lyrics[:train_index].to_numpy()
    lyrics_test = lyrics[train_index:].to_numpy()
    labels_train = labels[:train_index].to_numpy()
    labels_test = labels[train_index:].to_numpy()

    train_model_and_save(LR_MODEL_PATH, lyrics_train, labels_train, lyrics_test, labels_test)
    accuracy = load_model_and_test(LR_MODEL_PATH, lyrics_test, labels_test)
    print(f'{datetime.datetime.now()} - model accuracy after training - {accuracy}')


def test_word_average():
    lyrics = "hello hello hello"
    other_lyrics = "hello hello world"
    words = logistic_regresssion.lyrics_to_words(lyrics)
    other_words = logistic_regresssion.lyrics_to_words(other_lyrics)
    model = logistic_regresssion.LogisticRegressionClassifier()
    word_average_lyrics = model.song_lyrics_to_word_average(words)
    word_average_other_lyrics = model.song_lyrics_to_word_average(other_words)
    word_average_word = model.song_lyrics_to_word_average(["hello"])
    word_encoding = model.word_to_glove("hello")[0]

    diff_lyrics_word_avg = (word_average_lyrics - word_average_word).sum()
    diff_lyrics_word = (word_average_lyrics - word_encoding).sum()
    diff_lyrics_other = (word_average_lyrics - word_average_other_lyrics).sum()
    if (abs(diff_lyrics_word_avg) > 0.1):
        print(f"Abnormal diff between lyrics and single word avg - {diff_lyrics_word_avg}")
    if (abs(diff_lyrics_word) > 0.1):
        print(f"Abnormal diff between lyrics and single word encoding - {diff_lyrics_word}")
    if (abs(diff_lyrics_other) < 0.1):
        print(f"Abnormal low diff between lyrics and other lyrics - {diff_lyrics_other}")
    print("test over")

# test_word_average()

test_logistic_regression()

# test_majority_classifier()
