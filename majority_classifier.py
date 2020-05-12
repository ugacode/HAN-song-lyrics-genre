import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_metadata import DatasetMetadata


def test_network(model, test_dataset):
    with torch.no_grad():
        data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        correct_predictions = 0
        for _, sample_batched in enumerate(data_loader):
            prediction = model(sample_batched['lyrics'])
            if (prediction == sample_batched['genre']):
                correct_predictions += 1
        accuracy = (correct_predictions / len(data_loader)) * 100
        return accuracy


class MajorityClassifier(nn.Module):
    def __init__(self, metadata_file_path):
        super(MajorityClassifier, self).__init__()
        dataset_metadata = DatasetMetadata.from_filepath(metadata_file_path)
        self.majority_genre = dataset_metadata.most_common_genre_id

    def forward(self, x):
        return self.majority_genre
