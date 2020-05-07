import torch.nn as nn

from dataset_metadata import DatasetMetadata


class MajorityClassifier(nn.Module):
    def __init__(self, metadata_file_path):
        super(MajorityClassifier, self).__init__()
        dataset_metadata = DatasetMetadata.from_filepath(metadata_file_path)
        self.majority_genre = dataset_metadata.most_common_genre_str()

    def forward(self, x):
        return self.majority_genre
