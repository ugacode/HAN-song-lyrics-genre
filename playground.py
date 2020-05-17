import datetime

import majority_classifier
from dataset_loader import LyricsDataset, WordAverageTransform, LyricsDatasetEmbeddedHAN
import dataset_loader
from dataset_metadata import DatasetMetadata, JSON_FILE_PATH
import hierarchical_attention_net
from hierarchical_attention_net import HAN
from learning_dataset_generator import LEARNING_DATASET_TRAIN_PATH, LEARNING_DATASET_TEST_PATH, \
    LEARNING_SMALL_DATASET_TRAIN_PATH, LEARNING_TINY_DATASET_TRAIN_PATH, LEARNING_TINY_DATASET_TEST_PATH, \
    LEARNING_SMALL_DATASET_TEST_PATH
from majority_classifier import MajorityClassifier
import logistic_regresssion
import matplotlib.pyplot as plt
import pandas as pd
import csv

import torch

from plot_utils import plot_confusion_matrix

WORD_EMBEDDING_PATH = '.\\glove\\glove.6B.100d.txt'
LR_MODEL_PATH = '.\\models\\logistic_regression.model'
HAN_MODEL_PATH = '.\\models\\han.model'


def write_dataset_metadata():
    dm = DatasetMetadata()
    dm.dump_to_file(JSON_FILE_PATH)


def train_HAN_and_save(model_path, train_dataset, test_dataset):
    word_embedding = pd.read_csv(WORD_EMBEDDING_PATH, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    model = HAN(hierarchical_attention_net.HIDDEN_SIZE, hierarchical_attention_net.HIDDEN_SIZE,
                hierarchical_attention_net.BATCH_SIZE, hierarchical_attention_net.NUM_CLASSES,
                word_embedding)

    accuracy = hierarchical_attention_net.test_network(model, test_dataset)
    print(f'{datetime.datetime.now()} - model accuracy before training - {accuracy}')
    model = hierarchical_attention_net.train_network(model, train_dataset, 2)
    print(f'{datetime.datetime.now()} - model finished training')
    torch.save(model.state_dict(), model_path)


def load_HAN_and_test(model_path, test_dataset):
    word_embedding = pd.read_csv(WORD_EMBEDDING_PATH, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    model = HAN(hierarchical_attention_net.HIDDEN_SIZE, hierarchical_attention_net.HIDDEN_SIZE,
                hierarchical_attention_net.BATCH_SIZE, hierarchical_attention_net.NUM_CLASSES,
                word_embedding)
    model.load_state_dict(torch.load(model_path))
    accuracy, confusion = hierarchical_attention_net.test_network(model, test_dataset)

    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(confusion, dm.genre_labels, title="HAN Confusion Matrix")
    plt.show()
    return accuracy


def test_HAN(tiny=True):
    test_dataset_path = LEARNING_SMALL_DATASET_TEST_PATH
    train_dataset_path = LEARNING_DATASET_TRAIN_PATH
    if(tiny):
        test_dataset_path = LEARNING_TINY_DATASET_TEST_PATH
        train_dataset_path = LEARNING_TINY_DATASET_TRAIN_PATH

    print(f'{datetime.datetime.now()} - starting HAN testing')
    train_dataset = LyricsDatasetEmbeddedHAN(train_dataset_path, WORD_EMBEDDING_PATH)
    print(f'{datetime.datetime.now()} - loaded training data')
    test_dataset = LyricsDatasetEmbeddedHAN(test_dataset_path, WORD_EMBEDDING_PATH)
    print(f'{datetime.datetime.now()} - loaded testing data')

    train_HAN_and_save(HAN_MODEL_PATH, train_dataset, test_dataset)
    accuracy = load_HAN_and_test(HAN_MODEL_PATH, test_dataset)
    print(f'{datetime.datetime.now()} - model accuracy after training - {accuracy}')


def test_majority_classifier():
    print(f'{datetime.datetime.now()} - starting majority classifier testing')
    classifier = MajorityClassifier(JSON_FILE_PATH)
    test_dataset = LyricsDataset(LEARNING_DATASET_TEST_PATH, WordAverageTransform())
    accuracy, confusion = majority_classifier.test_network(classifier, test_dataset)

    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(confusion, dm.genre_labels, title="Majority Classifier Confusion Matrix")
    plt.show()

    print(f'{datetime.datetime.now()} - Majority classifier accuracy = {accuracy}')


def train_LR_and_save(model_path, train_dataset, test_dataset):
    model = logistic_regresssion.LogisticRegressionClassifier()
    accuracy, confusion = logistic_regresssion.test_network(model, test_dataset)
    print(f'{datetime.datetime.now()} - model accuracy before training - {accuracy}')
    model = logistic_regresssion.train_network(model, train_dataset, 10)
    print(f'{datetime.datetime.now()} - model finished training')
    torch.save(model.state_dict(), model_path)


def load_LR_and_test(model_path, test_dataset):

    model = logistic_regresssion.LogisticRegressionClassifier()
    model.load_state_dict(torch.load(model_path))
    accuracy, confusion = logistic_regresssion.test_network(model, test_dataset)

    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(confusion, dm.genre_labels, title="Logistic Regression Confusion Matrix")
    plt.show()
    return accuracy


def test_logistic_regression():
    print(f'{datetime.datetime.now()} - starting logistic regression testing')
    #train_dataset = LyricsDataset(LEARNING_SMALL_DATASET_TRAIN_PATH, WordAverageTransform())
    train_dataset = LyricsDataset(LEARNING_DATASET_TRAIN_PATH, WordAverageTransform())
    print(f'{datetime.datetime.now()} - loaded transformed training data')
    test_dataset = LyricsDataset(LEARNING_DATASET_TEST_PATH, WordAverageTransform())
    print(f'{datetime.datetime.now()} - loaded transformed testing data')

    train_LR_and_save(LR_MODEL_PATH, train_dataset, test_dataset)
    accuracy = load_LR_and_test(LR_MODEL_PATH, test_dataset)
    print(f'{datetime.datetime.now()} - model accuracy after training - {accuracy}')


# test_logistic_regression()

# test_majority_classifier()

test_HAN(False)
