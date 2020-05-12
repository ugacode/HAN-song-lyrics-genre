import pandas as pd

from dataset_metadata import DatasetMetadata, JSON_FILE_PATH

FULL_DATASET_PATH = '..\\lyrics\\dataset_clean.csv'
LEARNING_DATASET_TRAIN_PATH = '..\\lyrics\\learn_train.csv'
LEARNING_DATASET_TEST_PATH = '..\\lyrics\\learn_test.csv'
LEARNING_SMALL_DATASET_TRAIN_PATH = '..\\lyrics\\learn_small_train.csv'


def generate_learning_dataset():
    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    replace_dict = {k: v for v, k in enumerate(dm.genre_labels)}
    file = pd.read_csv(FULL_DATASET_PATH)
    field_drop = file.drop(columns=['index', 'song', 'artist'])
    genre_transform = field_drop.replace({'genre': replace_dict})
    shuffled = genre_transform.sample(frac=1)
    dataset_size = genre_transform.shape[0]
    train_index = int(dataset_size * 0.75)
    shuffled.head(train_index).to_csv(LEARNING_DATASET_TRAIN_PATH, index=False)
    shuffled.tail(dataset_size-train_index).to_csv(LEARNING_DATASET_TEST_PATH, index=False)
    shuffled.head(int(train_index * 0.4)).to_csv(LEARNING_SMALL_DATASET_TRAIN_PATH, index=False)

generate_learning_dataset()
