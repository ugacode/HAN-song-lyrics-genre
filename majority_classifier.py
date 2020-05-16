import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, ConfusionMatrix

from dataset_metadata import DatasetMetadata

BATCH_SIZE = 128


def test_network(model, test_dataset):
    with torch.no_grad():
        evaluator = create_supervised_evaluator(
            model, metrics={"accuracy": Accuracy(), "confusion": ConfusionMatrix(10)})
        data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        evaluator.run(data_loader)
        return evaluator.state.metrics["accuracy"] * 100, evaluator.state.metrics["confusion"]


class MajorityClassifier(nn.Module):
    def __init__(self, metadata_file_path):
        super(MajorityClassifier, self).__init__()
        self.dataset_metadata = DatasetMetadata.from_filepath(metadata_file_path)
        self.majority_genre = self.dataset_metadata.most_common_genre_id
        self.majority_genre_hot = torch.zeros(len(self.dataset_metadata.genre_labels))
        self.majority_genre_hot[self.majority_genre] = 1

    def forward(self, x):
        batch_size = x.shape[0]
        return self.majority_genre_hot.repeat(batch_size, 1)
