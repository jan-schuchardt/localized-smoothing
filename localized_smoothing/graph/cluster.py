from torch_geometric.loader import ClusterData
from torch_geometric.data import Data
import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import SpectralClustering


def spectral_clustering(n_clusters, adj, cluster_args):
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    **cluster_args).fit(adj)
    labels = clustering.labels_
    aff = clustering.affinity_matrix_
    ranking = {}
    for c in range(n_clusters):
        affinity = np.array([
            np.mean(aff[labels == c][:, labels == i]) for i in range(n_clusters)
        ])
        ranking[c] = list(affinity.argsort()[::-1])
    return ranking, labels


def metis_clustering(n_clusters, adj, cluster_args):
    edge_index = torch.tensor(np.array(
        (adj.nonzero()[0], adj.nonzero()[1]))).long()
    gd = Data(edge_index=edge_index)
    clustering = ClusterData(data=gd, num_parts=n_clusters, log=True)

    clabels = torch.ones(len(clustering.perm)) * -1
    n = len(clustering.perm)
    for ptr in range(len(clustering.partptr) - 1):
        for i in range(n):
            if (clustering.partptr[ptr] <= i) & (i
                                                 < clustering.partptr[ptr + 1]):
                clabels[clustering.perm[i]] = ptr

    # create ranking
    ranking = {}
    for c_from in range(n_clusters):
        current_ranking = []
        for c_to in range(n_clusters):
            num_edges = adj[clabels == c_from, :][:, clabels == c_to].sum()
            current_ranking.append(num_edges)
        ranking[c_from] = list(np.argsort(current_ranking)[::-1])
    return ranking, clabels


cluster_method = {
    'spectral_clustering': spectral_clustering,
    'metis': metis_clustering
}


class Cluster:

    def __init__(self, type, n_clusters, adj, cluster_args={}) -> None:
        self.n_clusters = n_clusters
        self._ranking, self._labels = self._cluster(type, n_clusters, adj,
                                                    cluster_args)
        self.nodes_per_cluster = np.bincount(self._labels)
        self._cluster_idxs = None
        self._rank_matrix = self._calc_rank_matrix()

    def get_cluster_idx(self, cluster) -> torch.Tensor:
        return self._cluster_idxs[cluster, :]

    def calc_cluster_indices(self, num_nodes) -> None:
        cluster_idxs = torch.empty(self.n_clusters, num_nodes, dtype=torch.long)
        for cluster in range(self.n_clusters):
            cluster_idxs[cluster, :] = self._calc_one_cluster_idx(
                cluster, list(range(num_nodes)))
        self._cluster_idxs = cluster_idxs

    def _calc_one_cluster_idx(self, cluster, nodes) -> torch.Tensor:
        """
        Returning the indices of attr_idx which belong to the cluster 'cluster'

        Parameters
        ----------
        cluster : int
            Cluster that is queried
        attr_idx : torch.Tensor [2, ?]
            The indices of the non-zero attributes

        Returns
        -------
        torch.Tensor [?]
            indices of the attributes
        """
        in_cluster = []
        for node in nodes:
            if self._labels[node] == cluster:
                in_cluster.append(True)
            else:
                in_cluster.append(False)
        return torch.tensor(in_cluster)

    def get_rank(self, c1, c2):
        return self._ranking[c1].index(c2)

    def _cluster(self, type, n_clusters, adj, cluster_args):
        return cluster_method[type](n_clusters, adj, cluster_args)

    @property
    def rank_matrix(self):
        """ A matrix providing the ranking for each cluster. E.g. row 1 is the ranking based 
        on the affinity from the point of view of cluster 1.
        Returns
        -------
        np.ndarray [n_cluster, n_clusters]
            Rank matrix
        """
        return self._rank_matrix

    def _calc_rank_matrix(self):
        rank_matrix = np.empty((self.n_clusters, self.n_clusters))
        for cluster in range(self.n_clusters):
            rank_matrix[cluster, :] = self._ranking[cluster]
        return rank_matrix.astype('int')
