import json

JSON_FILE_PATH = '.\\dataset_metadata.json'

class DatasetMetadata:
    def __init__(self, genre_labels=None, most_common_genre_id=0):
        if genre_labels is None:
            genre_labels = ["Pop", "Hip-Hop", "Rock", "Metal", "Country", "Jazz", "Electronic", "Folk", "R&B", "Indie"]
        self.genre_labels = genre_labels
        self.most_common_genre_id = most_common_genre_id

    def dump_to_file(self, file_path='.\\dataset_metadata.json'):
        out_dict = self.__dict__
        with open(file_path, 'w') as json_out:
            json.dump(out_dict, json_out)

    def most_common_genre_str(self):
        return self.genre_labels[self.most_common_genre_id]

    @classmethod
    def from_filepath(cls, file_path):
        with open(file_path, 'r') as json_in:
            return cls(**json.load(json_in))
