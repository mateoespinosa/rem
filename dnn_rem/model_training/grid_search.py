from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import itertools
import json
import logging
import tensorflow as tf

from .build_and_train_model import model_fn
from . import split_data


def load_best_params(best_params_file):
    indicator = " using "
    with open(best_params_file, 'r') as f:
        best_params_line = f.readline()
        indicator_ind = best_params_line.find(indicator)
        best_params_serialized = \
            best_params_line[indicator_ind + len(indicator):]
        return json.loads(best_params_serialized)


def grid_search(X, y, manager):
    """
    Args:
        X: input features
        y: target

    Returns:
        batch_size: best batch size
        epochs: best number of epochs
        layer_units: best number of neurons for hidden layers

    Perform a 5-folded grid search over the neural network hyper-parameters
    """

    # Make sure we DO NOT look into our test data when doing this grid search:
    X_train, y_train, _, _ = split_data.apply_split_indices(
        X=X,
        y=y,
        file_path=manager.NN_INIT_SPLIT_INDICES_FP,
        preprocess=manager.DATASET_INFO.preprocessing,
    )

    batch_size = [16, 32, 64, 128]
    epochs = [50, 100, 150, 200]
    layer_1 = [128, 64, 32, 16]
    layer_2 = [64, 32, 16, 8]

    param_grid = dict(
        input_features=[X.shape[-1]],
        num_outputs=[manager.DATASET_INFO.n_outputs],
        batch_size=batch_size,
        epochs=epochs,
        layer_units=list(itertools.product(layer_1, layer_2)),
    )

    model = KerasClassifier(build_fn=model_fn, verbose=0)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        cv=5,
        verbose=(
            int(logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        ),
    )
    grid_result = grid.fit(X_train, tf.keras.utils.to_categorical(y_train))

    # Write best results to file
    with open(manager.NN_INIT_GRID_RESULTS_FP, 'w') as file:
        file.write(
            f"Best: {grid_result.best_score_} using "
            f"{json.dumps(grid_result.best_params_)}\n"
        )

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            file.write(f"{mean} ({stdev}) with: {json.dumps(param)}\n")

    logging.debug('Grid Search for hyper parameters complete.')
    logging.debug(
        f"Best: {grid_result.best_score_} using {grid_result.best_params_}"
    )

    return grid_result.best_params_
