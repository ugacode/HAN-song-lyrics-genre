import pandas as pd

FULL_DATASET_PATH = '..\\lyrics\\lyrics.csv'
MOCK_DATASET_PATH = '..\\lyrics\\MOCK.csv'


def generate_mock_dataset():
    file = pd.read_csv(FULL_DATASET_PATH)
    file = file.dropna()
    is_aero = file['artist'] == 'aerosmith'
    aero_only = file[is_aero]
    field_drop = aero_only.drop(columns=['index', 'song', 'year', 'artist'])
    genre_transform = field_drop.replace({'genre': {'Rock': 2}})
    out_csv = genre_transform.to_csv(MOCK_DATASET_PATH, index=False)

# generate_mock_dataset()
