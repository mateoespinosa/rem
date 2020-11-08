"""
Generates neural networks for each of the n folds using the procedure
specified to locate optimal neural network hyper-parameters and neural network
initialization.

Used mostly for experimentation.
"""

from collections import OrderedDict
from tqdm import tqdm
import logging
import numpy as np
import os
import pandas as pd

from . import find_best_nn_initialisation
from .build_and_train_model import build_and_train_model
from .grid_search import (
    grid_search as grid_search_fn, load_best_params
)
from .split_data import load_split_indices
from . import split_data as split_data_fn


def run(
    X,
    y,
    manager,
    split_data=False,
    use_grid_search=False,
    find_best_initialisation=False,
    generate_fold_data=False,
):
    """
    Args:
        split_data: Split data. Only do this once!
        grid_search: Grid search to find best neural network
            hyper-parameters.
        find_best_initialisation: Find best neural network initialization
        generate_fold_data: Generate neural networks for each data fold

    """
    # 1. Split data into train and test. Only do this once
    if split_data:
        logging.warning('Splitting data.')
        split_data_fn.train_test_split(
            X=X,
            y=y,
            manager=manager,
            test_size=0.2,
        )
        split_data_fn.stratified_k_fold(
            X=X,
            y=y,
            manager=manager,
            n_folds=manager.N_FOLDS,
        )

    # 2. Grid search over neural network hyper params to find optimal neural
    #    network hyper-parameters
    if use_grid_search:
        grid_search_result = {}
        if os.path.exists(manager.NN_INIT_GRID_RESULTS_FP):
            # Then go ahead and extract the best hyper parameters for this
            # experiment out of this file.
            logging.warning(
                "Using previous best results from grid search found in "
                f"{manager.NN_INIT_GRID_RESULTS_FP}. This means we will "
                f"overwrite any provided hyper-parameters."
            )
            grid_search_result = load_best_params(
                manager.NN_INIT_GRID_RESULTS_FP
            )
        else:
            logging.warning(
                'Performing grid search over hyper parameters from scratch. '
                'This will take a while...'
            )
            grid_search_result = grid_search_fn(X=X, y=y, manager=manager)

        # And overwrite our given config's hyperparameters so that we use the
        # ones we found as best hyper-parameters for this experiment
        for hyper_param, value in grid_search_result.items():
            if hyper_param in manager.HYPERPARAMS:
                manager.HYPERPARAMS[hyper_param] = value

    # TODO change this to read best grid search hyper-parameters from disk

    # 3. Initialize 5 neural networks using 1 train test split
    # Pick initialization that yields the smallest ruleset
    if find_best_initialisation:
        logging.info("Finding best initialisation")
        find_best_nn_initialisation.run(X=X, y=y, manager=manager)

    # 4. Build neural network for each fold using best initialization found
    #    above
    if generate_fold_data:
        with tqdm(
            range(manager.N_FOLDS),
            desc="Training fold models"
        ) as pbar:
            for fold in pbar:
                pbar.set_description(
                    f'Training fold model {fold + 1}/{manager.N_FOLDS}'
                )

                # Split data using precomputed split indices
                train_index, test_index = load_split_indices(
                    manager.N_FOLD_CV_SPLIT_INDICIES_FP,
                    fold_index=fold,
                )
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                np.save(
                    manager.N_FOLD_CV_SPLIT_X_train_data_FP(fold),
                    X_train
                )
                np.save(
                    manager.N_FOLD_CV_SPLIT_y_train_data_FP(fold),
                    y_train,
                )
                np.save(
                    manager.N_FOLD_CV_SPLIT_X_test_data_FP(fold),
                    X_test,
                )
                np.save(
                    manager.N_FOLD_CV_SPLIT_y_test_data_FP(fold),
                    y_test,
                )

                # Model to be stored in
                # <dataset name>\cross_validation\<n>_folds\trained_models\
                model_file_path = manager.n_fold_model_fp(fold)
                acc, auc = build_and_train_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    manager=manager,
                    model_file_path=model_file_path,
                    with_best_initilisation=find_best_initialisation,
                )
                if logging.getLogger().getEffectiveLevel() not in [
                    logging.WARNING,
                    logging.ERROR
                ]:
                    pbar.write(
                        f'Test accuracy for fold {fold + 1}/{manager.N_FOLDS} '
                        f'is {acc} and AUC is {auc}'
                    )
