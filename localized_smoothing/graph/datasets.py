import torch
import numpy as np
from torch.utils.data import Dataset
from localized_smoothing.graph.utils import load_and_standardize


def get_dataset(name):
    graph = load_and_standardize('data/graphs/' + name + '.npz')
    return graph


class GraphDataset(Dataset):

    def __init__(self, dataset_name, indices) -> None:
        super(GraphDataset, self).__init__()
        graph = get_dataset(dataset_name)
        self.n, self.d = graph.attr_matrix.shape
        self.nc = graph.labels.max() + 1
        self.edge_idx = torch.LongTensor(np.stack(graph.adj_matrix.nonzero()))
        self.attr_idx = torch.LongTensor(np.stack(graph.attr_matrix.nonzero()))
        self.labels = graph.labels
        self.indices = indices

    def __getitem__(self, index):
        return self.attr_idx, self.edge_idx, self.labels[
            self.indices], self.indices

    def __len__(self):
        return 1
