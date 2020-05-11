from dataset_metadata import DatasetMetadata
from majority_classifier import MajorityClassifier
import logistic_regresssion

import torch

import pandas as pd

JSON_FILE_PATH = '.\\dataset_metadata.json'
MOCK_DATASET_PATH = '..\\lyrics\\MOCK.csv'
LR_MODEL_PATH = '.\\models\\logistic_regression'


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
    print(f'model accuracy before training - {accuracy}')
    word_average_train_data = [model.song_lyrics_to_word_average(song) for song in train_data]
    word_average_train_data_combined = torch.stack(word_average_train_data, dim=0)
    model = logistic_regresssion.train_network(model, word_average_train_data_combined, train_labels, 10)
    torch.save(model.state_dict(), model_path)


def load_model_and_test(model_path, test_data, test_labels):
    model = logistic_regresssion.LogisticRegressionClassifier()
    model.load_state_dict(torch.load(model_path))
    accuracy = logistic_regresssion.test_network(model, test_data, test_labels)
    return accuracy


def test_logistic_regerssion():
    mock_data = pd.read_csv(MOCK_DATASET_PATH)
    data_size = mock_data.shape[0]
    train_index = int(data_size * 0.9)
    # split into labels and lyrics
    lyrics = mock_data['lyrics'].transform(logistic_regresssion.lyrics_to_words)
    labels = mock_data['genre']
    lyrics_train = lyrics[:train_index].to_numpy()
    lyrics_test = lyrics[train_index:].to_numpy()
    labels_train = labels[:train_index].to_numpy()
    labels_test = labels[train_index:].to_numpy()

    train_model_and_save(LR_MODEL_PATH, lyrics_train, labels_train, lyrics_test, labels_test)
    accuracy = load_model_and_test(LR_MODEL_PATH, lyrics_test, labels_test)
    print(f'model accuracy after training - {accuracy}')

#test_logistic_regerssion()

#test_majority_classifier()
