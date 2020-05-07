from dataset_metadata import DatasetMetadata
from majority_classifier import MajorityClassifier

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


test_majority_classifier()
