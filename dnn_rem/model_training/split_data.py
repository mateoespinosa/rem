"""
Split data and save indices of the split for reproducibility
"""
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
import logging
import os
import pandas as pd
import pathlib


def save_split_indices(train_indices, test_indices, file_path):
    """
    Args:
        train_indices: List of train indices of data
        test_indices: List of test indices of data
        file_path: File to save split indices

    Write list of indices to split_indices.txt

    File of the form with a train and test line for each fold
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...

    """
    with open(file_path, 'a') as file:
        file.write(
            'train ' + ' '.join([str(index) for index in train_indices]) + '\n'
        )
        file.write(
            'test ' + ' '.join([str(index) for index in test_indices]) + '\n'
        )


def apply_split_indices(
    X,
    y,
    file_path,
    preprocess=None,
    fold_index=0,
):
    """
    Args:
        file_path: path to split indices file
        fold_index: index of the fold whose train and test indices you want

    Returns:
        train_indices: list of integer indices for train data
        test_indices: list of integer indices for test data

    File of the form
    train 0 1 0 2 ...
    test 3 4 6 ...
    train 1 5 2 ...
    test 6 8 9 ...
    ...
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) >= (2 * fold_index) + 2, (
            f'Error: not enough information in fold indices file '
            f'{len(lines)} < {(2 * fold_index)}'
        )
        train_indices = lines[(fold_index * 2)].split(' ')[1:]
        test_indices = lines[(fold_index * 2) + 1].split(' ')[1:]

    # Convert string indices to ints
    train_indices = list(map(int, train_indices))
    test_indices = list(map(int, test_indices))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # And do any required preprocessing we may need to do to the
    # data now that it has been partitioned (to avoid information
    # leakage in data-dependent preprocessing passes)
    if preprocess:
        X_train, y_train = preprocess(X_train, y_train)
        X_test, y_test = preprocess(X_test, y_test)
    return X_train, y_train, X_test, y_test


def stratified_k_fold(X, y, n_folds, manager):
    """
    Args:
        X: input features
        y: target
        n_folds: how many folds to split data into

    Split data into folds and saves indices in to data_split_indices.txt
    """

    # Make directory for
    # Create directory: cross_validation/<n>_folds/
    os.makedirs(manager.N_FOLD_CV_DP, exist_ok=True)
    # Create directory: <n>_folds/rule_extraction/<rulemode>/
    os.makedirs(manager.N_FOLD_RULE_EX_MODE_DP, exist_ok=True)
    # Create directory: <n>_folds/rule_extraction/<rulemode>/rules_extracted
    os.makedirs(manager.N_FOLD_RULES_DP, exist_ok=True)
    # Create directory: <n>_folds/trained_models
    os.makedirs(manager.N_FOLD_MODELS_DP, exist_ok=True)

    # Initialise split indices file
    os.makedirs(
        pathlib.Path(manager.N_FOLD_CV_SPLIT_INDICIES_FP).parent,
        exist_ok=True,
    )
    open(manager.N_FOLD_CV_SPLIT_INDICIES_FP, 'w').close()
    if n_folds == 1:
        # Degenerate case: let's just dump all our indices as our single fold
        partition = ShuffleSplit(
            n_splits=1,
            test_size=manager.PERCENT_TEST_DATA,
            random_state=42,
        )
        train_indices, test_indices = next(partition.split(X, y))
        save_split_indices(
            train_indices=train_indices,
            test_indices=test_indices,
            file_path=manager.N_FOLD_CV_SPLIT_INDICIES_FP,
        )
        return

    # Split data
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )

    # Save indices
    for train_indices, test_indices in skf.split(X, y):
        save_split_indices(
            train_indices=train_indices,
            test_indices=test_indices,
            file_path=manager.N_FOLD_CV_SPLIT_INDICIES_FP,
        )

    logging.debug(f'Split data into {n_folds} folds.')


def train_test_split(X, y, manager, test_size=0.2):
    """

    Args:
        X: input features
        y: target
        test_size: percentage of the data used for testing

    Returns:

    Single train test split of the data used for initilising the neural network
    """

    # Initialise split indices file
    os.makedirs(
        pathlib.Path(manager.NN_INIT_SPLIT_INDICES_FP).parent,
        exist_ok=True,
    )
    open(manager.NN_INIT_SPLIT_INDICES_FP, 'w').close()

    # Split data
    rs = ShuffleSplit(
        n_splits=2,
        test_size=test_size,
        random_state=42,
    )

    for train_indices, test_indices in rs.split(X):
        save_split_indices(
            train_indices=train_indices,
            test_indices=test_indices,
            file_path=manager.NN_INIT_SPLIT_INDICES_FP,
        )

        # Only want 1 split
        break

    logging.debug('Split data into train/test split for initialisation.')


def load_data(dataset_info, data_path):
    """

    Args:
        dataset_info: meta data about dataset e.g. name, target col
        data_path: path to data.csv

    Returns:
        X: data input features
        y: data target
    """
    data = pd.read_csv(data_path)

    X = data.drop([dataset_info.target_col], axis=1).values
    y = data[dataset_info.target_col].values

    return X, y


def feature_names(data_path):
    data = pd.read_csv(data_path)
    return list(data.columns)
