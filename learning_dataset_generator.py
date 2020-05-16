import pandas as pd

from dataset_metadata import DatasetMetadata, JSON_FILE_PATH

FULL_DATASET_PATH = '..\\lyrics\\dataset_clean.csv'
LEARNING_DATASET_TRAIN_PATH = '..\\lyrics\\learn_train.csv'
LEARNING_DATASET_TEST_PATH = '..\\lyrics\\learn_test.csv'
LEARNING_SMALL_DATASET_TEST_PATH = '..\\lyrics\\learn_small_test.csv'
LEARNING_TINY_DATASET_TEST_PATH = '..\\lyrics\\learn_tiny_test.csv'
LEARNING_SMALL_DATASET_TRAIN_PATH = '..\\lyrics\\learn_small_train.csv'
LEARNING_TINY_DATASET_TRAIN_PATH = '..\\lyrics\\learn_tiny_train.csv'


def generate_learning_dataset():
    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    replace_dict = {k: v for v, k in enumerate(dm.genre_labels)}
    file = pd.read_csv(FULL_DATASET_PATH)
    field_drop = file.drop(columns=['index'])
    genre_transform = field_drop.replace({'genre': replace_dict})
    undersample_list = []
    for genre_i in range(len(dm.genre_labels)):
        genre_only = genre_transform.loc[genre_transform["genre"] == genre_i]
        if (len(genre_only) > 15591):
            genre_only = genre_only.sample(15591)
        undersample_list.append(genre_only)

    undersampled = pd.concat(undersample_list, axis=0)
    shuffled = undersampled.sample(frac=1)
    dataset_size = genre_transform.shape[0]
    train_index = int(dataset_size * 0.75)
    shuffled.head(train_index).to_csv(LEARNING_DATASET_TRAIN_PATH, index=False)

    shuffled.tail(dataset_size-train_index).to_csv(LEARNING_DATASET_TEST_PATH, index=False)
    shuffled.head(int(train_index * 0.4)).to_csv(LEARNING_SMALL_DATASET_TRAIN_PATH, index=False)
    shuffled.head(int(train_index * 0.02)).to_csv(LEARNING_TINY_DATASET_TRAIN_PATH, index=False)
    shuffled.tail(int((dataset_size - train_index) * 0.05)).to_csv(LEARNING_TINY_DATASET_TEST_PATH, index=False)
    shuffled.tail(int((dataset_size - train_index) * 0.25)).to_csv(LEARNING_SMALL_DATASET_TEST_PATH, index=False)

generate_learning_dataset()
