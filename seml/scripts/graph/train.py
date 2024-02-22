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
from localized_smoothing.graph.datasets import GraphDataset
from localized_smoothing.graph.trainer import ModelWrapper

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import ProgressBar

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
def run(experiment_name, dataset_name, model_data, n_clusters, training_params,
        certify_test, min_p, max_p, min_m, max_m, n_samples, seed):

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(size=8,
                     chars=string.ascii_uppercase + string.ascii_lowercase +
                     string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # Create directories
    # A unique directory name is created for this run based on the input
    directory = os.path.join(
        'experiments',
        experiment_name,
    )
    name = os.path.join(
        os.path.basename(dataset_name), model_data['name'],
        f'pp_min_{min_p}_pm_min_{min_m}' + '_lr_' +
        str(np.round(training_params['learning_rate'], 4)) + '_weight_decay_' +
        str(np.round(training_params['weight_decay'], 4)) + '_drop_' +
        str(np.round(model_data['model_params']['p_dropout'], 4)) + '_seed_' +
        str(seed))
    directory = os.path.join(directory, name)
    print(f'\n\n\n {directory} \n\n\n\n')
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
    trainset = GraphDataset(dataset_name, idx_train)
    valset = GraphDataset(dataset_name, indices)
    print(model_data)
    model = get_model(d=d, nc=nc, **model_data)

    L = graph.calc_laplacian()
    clustering = Cluster('spectral_clustering', n_clusters, graph.adj_matrix)
    clustering.calc_cluster_indices(num_nodes=graph.attr_matrix.shape[0])

    n_samples_eval = n_samples
    n_samples_pre_eval = 0
    pis = calc_pis(min_p=min_p,
                   max_p=max_p,
                   min_m=min_m,
                   max_m=max_m,
                   n_clusters=n_clusters)
    print(pis)
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

    lightning_model = ModelWrapper(model=smooth,
                                   trainset=trainset,
                                   valset=valset,
                                   **training_params)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(save_dir=os.path.abspath(directory),
                               name=name,
                               project='localized-smoothing-standardised')
    progress_bar = ProgressBar(refresh_rate=0)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{val_acc:.4f}', monitor='val_loss', save_top_k=1)
    from pytorch_lightning.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_acc', patience=1000, mode="max")
    trainer = pl.Trainer(logger=wandb_logger,
                         gpus=1,
                         max_epochs=3000,
                         callbacks=[
                             checkpoint_callback, lr_monitor, progress_bar,
                             early_stopping
                         ])
    trainer.fit(lightning_model)
    print(trainer.logged_metrics)
    return {
        key + '_best': trainer.logged_metrics[key]
        for key in trainer.logged_metrics
    }
