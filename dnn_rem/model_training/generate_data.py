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
import tensorflow as tf

from . import find_best_nn_initialisation
from .build_and_train_model import run_train_loop, load_model
from .grid_search import (
    grid_search as grid_search_fn, deserialize_best_params,
    serialize_best_params
)


def _save_model_and_return(model, path):
    # Helper method to serialize a given Keras model to a path and then
    # return the same model. Used for our checkpointing barriers.
    model.save(path)
    return model


def run(manager, use_grid_search=False):

    # 1. Grid search over neural network hyper params to find optimal neural
    #    network hyper-parameters
    if use_grid_search:
        X_train, y_train, _, _ = manager.get_train_split()
        grid_search_result, _ = manager.serializable_stage(
            target_file=manager.NN_INIT_GRID_RESULTS_FP,
            execute_fn=lambda: grid_search_fn(
                X=X_train,
                y=y_train,
                num_outputs=manager.DATASET_INFO.n_outputs,
            ),
            serializing_fn=serialize_best_params,
            deserializing_fn=deserialize_best_params,
        )

        # And overwrite our given config's hyperparameters so that we use the
        # ones we found as best hyper-parameters for this experiment
        for hyper_param, value in grid_search_result.items():
            if hyper_param in manager.HYPERPARAMS:
                manager.HYPERPARAMS[hyper_param] = value

    # 3. Initialize some neural networks using 1 train test split
    # Pick initialization that yields the smallest ruleset if we are asked by
    # the manager
    if manager.INITIALISATION_TRIALS > 1:
        manager.serializable_stage(
            target_file=manager.BEST_NN_INIT_FP,
            execute_fn=lambda: find_best_nn_initialisation.run(manager),
            serializing_fn=_save_model_and_return,
        )

    # 4. Build neural network for each fold using best initialization found
    #    above
    def _train_fold(fold, pbar):
        # Split data using precomputed split indices
        X_train, y_train, X_test, y_test = manager.get_fold_data(fold)

        # Actually build and train the model
        model, acc, auc, maj_class_acc = run_train_loop(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            manager=manager,
        )

        # Log to the progress bar if not on quiet mode
        if logging.getLogger().getEffectiveLevel() not in [
            logging.WARNING,
            logging.ERROR
        ]:
            pbar.write(
                f'Test accuracy for fold {fold + 1}/{manager.N_FOLDS} '
                f'is {round(acc, 3)}, AUC is {round(auc, 3)}, and '
                f'majority class accuracy is {round(maj_class_acc, 3)}'
            )
        return model

    # We instantiate a progress bar that keeps track how many more models
    # we will need to train
    with tqdm(
        range(manager.N_FOLDS),
        desc="Training fold models"
    ) as pbar:
        for fold in pbar:
            pbar.set_description(
                f'Training fold model {fold + 1}/{manager.N_FOLDS}'
            )
            # Manager bypass here if we have already trained a model for this
            # fold
            model, deserialized = manager.serializable_stage(
                target_file=manager.n_fold_model_fp(fold),
                execute_fn=lambda: _train_fold(fold, pbar),
                serializing_fn=_save_model_and_return,
                deserializing_fn=load_model,
            )
            if deserialized:
                # Then let's try and be nice and include some statistics
                # reporting for the performance of this model
                _, _, X_test, y_test = manager.get_fold_data(fold)
                _, auc, acc, maj_class_acc = model.evaluate(
                    X_test,
                    tf.keras.utils.to_categorical(y_test),
                    verbose=0,
                )
                pbar.write(
                    f'Test accuracy for fold {fold + 1}/{manager.N_FOLDS} '
                    f'is {round(acc, 3)}, AUC is {round(auc, 3)}, and '
                    f'majority class accuracy is {round(maj_class_acc, 3)}'
                )

