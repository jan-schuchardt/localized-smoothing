#!/usr/bin/env python3
import numpy as np
import os
import logging
import string
import random
from datetime import datetime
from localized_smoothing.graph.models import get_model
from localized_smoothing.graph.datasets import get_dataset
from localized_smoothing.graph.utils import *
from localized_smoothing.graph.cluster import Cluster
from localized_smoothing.graph.smooth import SmoothDiscrete
from localized_smoothing.graph.certification import certify

from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(experiment_name, dataset_name, model_data, budget_a, budget_d,
        n_clusters, certify_test, min_p, max_p, min_m, max_m, n_samples_eval,
        n_samples_pre_eval, cluster_args, batch_size, clustering_type, seed):

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(size=8,
                     chars=string.ascii_uppercase + string.ascii_lowercase +
                     string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # Create directories
    # A unique directory name is created for this run based on the input
    directory = 'experiments/' + experiment_name + "/" + datetime.now(
    ).strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
    logging.info(f"Directory: {directory}")
    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)

    graph = get_dataset(dataset_name)
    adj = graph.adj_matrix
    n, d = graph.attr_matrix.shape
    nc = graph.labels.max() + 1
    idx_train, idx_val, idx_test = split(graph.labels, seed=seed)
    indices = idx_test if certify_test else idx_val
    model = get_model(d=d, nc=nc, **model_data)

    clustering = Cluster(clustering_type,
                         n_clusters,
                         graph.adj_matrix,
                         cluster_args=cluster_args)
    clustering.calc_cluster_indices(num_nodes=graph.attr_matrix.shape[0])

    pis = calc_pis(min_p=min_p,
                   max_p=max_p,
                   min_m=min_m,
                   max_m=max_m,
                   n_clusters=n_clusters)

    print(n_samples_eval)
    sample_config = {
        'n_samples': n_samples_eval,
        'n_samples_pre_eval': n_samples_pre_eval,
        'pis': pis
    }
    smooth = SmoothDiscrete(
        graph=graph,
        model=model,
        clustering=clustering,
        sample_config=sample_config,
    )
    # eval model
    budgets_a = np.array(range(budget_a))
    budgets_d = np.array(range(budget_d))

    naive_results, collective_results, num_abstentions = certify(
        smooth,
        budgets_a,
        budgets_d,
        indices,
        graph.labels,
        is_mip=True,
        batch_size=batch_size)

    pretty(naive_results)
    pretty(collective_results)
    from localized_smoothing.graph.utils import save_obj

    save_obj(naive_results, directory, 'naive_results')
    save_obj(collective_results, directory, 'collective_results')
    return {
        'result_path': directory,
        'num_abstentions': num_abstentions,
        'naive_results': naive_results,
        'collective_results': collective_results
    }
