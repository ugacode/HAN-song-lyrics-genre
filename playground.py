from dataset_metadata import DatasetMetadata
from majority_classifier import MajorityClassifier
from logistic_regresssion import LogisticRegressionClassifier
import logistic_regresssion

import torch

JSON_FILE_PATH = '.\\dataset_metadata.json'


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


def train_model_and_save(model_path):
    # load train data
    train_data = None
    # load train labels
    train_labels = None

    model = logistic_regresssion.train_network(train_data, train_labels, 10)
    torch.save(model.state_dict(), model_path)


def load_model_and_test(model_path):
    # load test data
    test_data = None
    # load test labels
    test_labels = None

    model = logistic_regresssion.LogisticRegressionClassifier()
    model.load_state_dict(torch.load(model_path))
    accuracy = logistic_regresssion.test_network(model, test_data, test_labels)
    return accuracy


test_majority_classifier()
