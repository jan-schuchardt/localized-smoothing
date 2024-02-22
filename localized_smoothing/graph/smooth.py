from tqdm import tqdm
from math import ceil, sqrt, log
from typing import Tuple
from localized_smoothing.graph.utils import *
import torch.nn as nn
import numpy as np
from localized_smoothing.graph.cluster import Cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmoothDiscrete(nn.Module):

    def __init__(
        self,
        graph: SparseGraph,
        model: nn.Module,
        clustering: Cluster,
        sample_config: dict,
        num_thr=10000,
        conf_threshold=0.01,
    ):
        super(SmoothDiscrete, self).__init__()
        self.model = model.to(device)
        self.clustering = clustering
        self.sample_config = sample_config
        self.num_thr = num_thr
        self.confidence = conf_threshold
        self.a = 0
        self.b = 1

        self.n, self.d = graph.attr_matrix.shape
        self.nc = graph.labels.max() + 1
        self.edge_idx = torch.LongTensor(np.stack(
            graph.adj_matrix.nonzero())).to(device)
        self.attr_idx = torch.LongTensor(np.stack(
            graph.attr_matrix.nonzero())).to(device)

    def sample_noise(self,
                     batch_size,
                     attr_idx=None,
                     edge_idx=None,
                     sample_config=None):
        if sample_config is None:
            sample_config = self.sample_config

        if attr_idx is None:
            attr_idx = self.attr_idx
        if edge_idx is None:
            edge_idx = self.edge_idx

        return self._sample_noise(attr_idx=attr_idx,
                                  edge_idx=edge_idx,
                                  sample_config=sample_config,
                                  model=self.model,
                                  clustering=self.clustering,
                                  n=self.n,
                                  nc=self.nc,
                                  d=self.d,
                                  batch_size=batch_size)

    def forward(self, batch_size, attr_idx, edge_idx, sample_config=None):
        if sample_config is None:
            sample_config = self.sample_config

        return self._forward(attr_idx=attr_idx,
                             edge_idx=edge_idx,
                             sample_config=sample_config,
                             model=self.model,
                             clustering=self.clustering,
                             n=self.n,
                             nc=self.nc,
                             d=self.d,
                             batch_size=batch_size)

    def _forward(self,
                 attr_idx,
                 edge_idx,
                 sample_config,
                 model,
                 clustering,
                 n,
                 d,
                 nc,
                 batch_size=1,
                 grad_enabled=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        attr_idx: torch.Tensor [2, ?]
            The indices of the non-zero attributes.
        edge_idx: torch.Tensor [2, ?]
            The indices of the edges.
        sample_config: dict
            Configuration specifying the sampling probabilities
        model : torch.nn.Module
            The GNN model.
        n : int
            Number of nodes
        d : int
            Number of features
        nc : int
            Number of classes
        batch_size : int
            Number of graphs to sample per batch
        Returns
        -------
        votes : array-like [n, nc]
            The votes per class for each node
        scores: array-like [n_samples, n, nc]
            The scores of all samples for each node and class
        """
        n_samples = sample_config.get('n_samples', 1)
        pis = sample_config.get('pis')
        model.eval()
        votes = torch.zeros((n, nc), dtype=torch.float, device=attr_idx.device)
        score_device = torch.device("cuda" if grad_enabled else "cpu")
        saved_scores = torch.zeros((n_samples, n, nc),
                                   dtype=torch.float,
                                   device=score_device)
        with torch.set_grad_enabled(grad_enabled):
            assert n_samples % batch_size == 0
            nbatches = n_samples // batch_size
            for c in range(clustering.n_clusters):
                current_nodes = clustering.get_cluster_idx(c).bool()
                for i in tqdm(range(nbatches)):
                    attr_idx_batch, edge_idx_batch = self._sample_clustered_data(
                        attr_idx=attr_idx,
                        edge_idx=edge_idx,
                        clustering=clustering,
                        cluster=c,
                        pis=pis,
                        n=n,
                        d=d,
                        nsamples=batch_size)

                    scores = model(attr_idx=attr_idx_batch,
                                   edge_idx=edge_idx_batch,
                                   n=batch_size * n,
                                   d=d)
                    #predictions = scores.argmax(1)
                    predictions = scores.reshape(batch_size, n, nc).sum(0)
                    votes[current_nodes, :] += predictions[current_nodes, :]

                    saved_scores[i * batch_size:(i + 1) * batch_size,
                                 current_nodes, :] = scores.reshape(
                                     batch_size, n,
                                     nc)[:, current_nodes, :].to(score_device)
                    del attr_idx_batch
                    del edge_idx_batch
                    torch.cuda.empty_cache()

        return votes, saved_scores

    def _sample_noise(self,
                      attr_idx,
                      edge_idx,
                      sample_config,
                      model,
                      clustering,
                      n,
                      d,
                      nc,
                      batch_size=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        attr_idx: torch.Tensor [2, ?]
            The indices of the non-zero attributes.
        edge_idx: torch.Tensor [2, ?]
            The indices of the edges.
        sample_config: dict
            Configuration specifying the sampling probabilities
        model : torch.nn.Module
            The GNN model.
        n : int
            Number of nodes
        d : int
            Number of features
        nc : int
            Number of classes
        batch_size : int
            Number of graphs to sample per batch
        Returns
        -------
        votes : array-like [n, nc]
            The votes per class for each node
        scores: array-like [n_samples, n, nc]
            The scores of all samples for each node and class
        """
        n_samples = sample_config.get('n_samples', 1)
        votes, saved_scores = self._forward(attr_idx=attr_idx,
                                            edge_idx=edge_idx,
                                            sample_config=sample_config,
                                            model=model,
                                            clustering=clustering,
                                            n=n,
                                            d=d,
                                            nc=nc,
                                            batch_size=batch_size,
                                            grad_enabled=False)

        return (votes / n_samples), saved_scores

    # use the output of this function now in the prediction
    # we predict with it and store the predictions for all the nodes of cluster 'cluster'
    def _sample_clustered_data(self, clustering: Cluster, cluster, attr_idx,
                               edge_idx, pis, n, d, nsamples):
        """
        general idea: 
        cut the data indices in the specific parts for the clusters and pass them on 
        with their specific sampling probabilities perturb them in the end build them back together 
        in a larger list (all at the correct positions to get the predictions correct for the output node)
        
        Parameters
        ----------
        clustering: Clustering
            Clustering object containing a clustering of the graph
        cluster: int
            The cluster for which we want to sample
        attr_idx: torch.Tensor [2, ?]
            The indices of the non-zero attributes
        edge_idx: torch.Tensor [2, ?]
            The indices of the edges
        pis: List[Tuple[float, float]]
            Sampling probabilities for zeroes and ones in a tuple for the corresponding cluster. 
            They are sorted in an increasing way. pi's with larger values should be used for clusters
            with lower affinity
        n: int
            Number of nodes
        d: int
            Number of features
        nsamples: int
            Number of samples
        
        Returns
        -------
        something
        """
        per_attr_idx = []
        # now sample the attributes differently, edges can be done only copied once
        for c in range(clustering.n_clusters):
            # get p+ and p-
            p_p, p_m = pis[clustering.get_rank(cluster, c)]
            # get idx of first cluster
            cluster_idx = clustering.get_cluster_idx(c).bool()
            # perturb the attributes with the corresponding p+ and p-
            c_per_attr_idx = sample_multiple_graphs_attr(attr_idx=attr_idx,
                                                         pf_plus_att=p_p,
                                                         pf_minus_att=p_m,
                                                         n=n,
                                                         d=d,
                                                         nsamples=nsamples)
            mask = cluster_idx[c_per_attr_idx[0] % n]
            per_attr_idx.append(c_per_attr_idx[:, mask])
        # get copied edges
        per_edge_idx = sample_multiple_graphs_edges(edge_idx,
                                                    n=n,
                                                    nsamples=nsamples)
        per_attr_idx = torch.cat(per_attr_idx, dim=1)
        per_attr_idx_sorted = per_attr_idx[:, per_attr_idx[0].sort()[1]]
        return per_attr_idx_sorted, per_edge_idx

    def sort_scores(self, scores: np.ndarray) -> np.ndarray:
        return np.sort(scores, axis=0)

    def bin_thresholds(self, cdf_scores: np.ndarray) -> np.ndarray:
        """Groups the cdf scores in equally spaced bins

        Parameters
        ----------
        cdf_scores : np.ndarray
            CDF scores we get from predicting the perturbed input

        Returns
        -------
        np.ndarray
            Binned CDF scores called thresholds in the following
        """
        n_samples = self.sample_config.get('n_samples')
        gap = ceil(n_samples / self.num_thr)
        thresholds = cdf_scores[::gap]
        a = np.ones(thresholds.shape[1]) * self.a
        b = np.ones_like(a) * self.b
        thresholds = np.vstack((a, thresholds, b))
        return thresholds

    def calc_eps(self) -> float:
        """Calculates the confidence bound around the CDF

        Returns
        -------
        float
            Confidence bound epsilon
        """
        n_samples = self.sample_config.get('n_samples')
        return sqrt(log(2 / self.confidence) / (2 * n_samples))

    def estimate_bounds(
            self, cdf_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computing the mean and variance bounds based on the empirical distribution

        Parameters
        ----------
        cdf_scores : np.ndarray
            Confidence scores of the output

        Returns
        -------
        Tuple[np.ndarray, np.ndarray] [num_nodes, n_classes], [num_nodes, n_classes]
            Mean bounds and variance bounds per class
        """
        eps = self.calc_eps()
        cdf_scores = self.sort_scores(cdf_scores)
        thresholds = self.bin_thresholds(cdf_scores)
        mean = self.conf_interval_mean(
            thresholds, 0)[0]  # with zero bound we get empiric mean
        mean_bounds = self.conf_interval_mean(
            thresholds, eps)[0]  #TODO: check if upper oder lower
        var_bounds = self.conf_interval_variance(
            thresholds, eps, mean)[1]  #we can also use the bound
        return mean_bounds, var_bounds

    def conf_interval_mean(self, thresholds: np.ndarray, eps: float):
        """Estimating the empiric confidence interval on the mean given the samples.

        Parameters
        ----------
        eps : float
            The bound on the perturbation of the input
        cdf_scores : np.ndarray
            Threshold on the confidence scores from the samples
        statistical_confidence : [type]
            Statistical confidence bound on the CDF scores

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            vector of lower bounds and vector of upper bounds of the mean
        """
        n_bins = len(thresholds) - 2
        ps_upper = np.minimum(np.linspace(1, 1 / n_bins, n_bins) + eps, 1.0)
        upper = thresholds[1] + np.diff(thresholds[1:], axis=0).T @ ps_upper

        ps_lower = np.maximum(np.linspace(1, 1 / n_bins, n_bins) - eps, 0.0)
        lower = thresholds[0] + np.diff(thresholds[:-1], axis=0).T @ ps_lower

        return lower, upper

    def conf_interval_variance(self, thresholds: np.ndarray, eps: float,
                               mean: float):
        n_bins = len(thresholds) - 2
        a = np.ones(thresholds.shape[1]) * self.a
        b = np.ones_like(a) * self.b
        intervals = np.dstack(
            (thresholds[:-1], thresholds[1:]))

        ps_upper = np.minimum(np.linspace(1, 1 / n_bins, n_bins) + eps, 1.0)
        ps_upper = np.repeat(ps_upper[:, np.newaxis], len(a), axis=1)
        ps_lower = np.maximum(np.linspace(1, 1 / n_bins, n_bins) - eps, 0.0)
        ps_lower = np.repeat(ps_lower[:, np.newaxis], len(a), axis=1)

        max_l2 = np.max((intervals - mean[None, :, None])**2,
                        axis=2)
        l2_diff = np.diff(max_l2, axis=0)
        weight = np.empty_like(l2_diff)

        weight[l2_diff < 0] = ps_lower[l2_diff < 0]
        upper = max_l2[0] + (l2_diff * weight).sum(axis=0)

        min_l2 = np.min((intervals - mean[None, :, None])**2, axis=2)
        l2_diff = np.diff(min_l2, axis=0)
        weight = np.empty_like(l2_diff)
        weight[l2_diff >= 0] = ps_lower[l2_diff >= 0]
        weight[l2_diff < 0] = ps_upper[l2_diff < 0]
        lower = min_l2[0] + (l2_diff * weight).sum(axis=0)
        return lower, upper
