import json


class DatasetMetadata:
    def __init__(self, genre_labels=None, most_common_genre_id=0, song_count=100):
        if genre_labels is None:
            genre_labels = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6', 'genre_7',
                            'genre_8', 'genre_9', 'genre_10']
        self.genre_labels = genre_labels
        self.most_common_genre_id = most_common_genre_id
        self.song_count = song_count

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
