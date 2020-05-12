import pandas as pd

from dataset_metadata import DatasetMetadata, JSON_FILE_PATH

FULL_DATASET_PATH = '..\\lyrics\\dataset_clean.csv'
MOCK_DATASET_PATH = '..\\lyrics\\MOCK.csv'


def generate_mock_dataset():
    dm = DatasetMetadata.from_filepath(JSON_FILE_PATH)
    replace_dict = {k: v for v, k in enumerate(dm.genre_labels)}
    file = pd.read_csv(FULL_DATASET_PATH)
    field_drop = file.drop(columns=['index', 'song', 'artist'])
    genre_transform = field_drop.replace({'genre': replace_dict})
    #genre_transform.head(80000).to_csv(MOCK_DATASET_PATH, index=False)
    genre_transform.to_csv(MOCK_DATASET_PATH, index=False)


generate_mock_dataset()
