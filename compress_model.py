#!/usr/bin/env python3

"""
Executable tool for compressing a given TF model serialized as an h5 file and
dumping the new compressed model into the given output path.
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable
import argparse
import logging
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
import warnings

from dnn_rem.experiment_runners import dataset_configs
from dnn_rem.utils.resources import resource_compute
from dnn_rem.compression.vanilla_magnitude_compress import \
    weight_magnitude_compress, neuron_magnitude_compress
from dnn_rem.model_training.build_and_train_model import load_model

_RANDOM_SEED = 42

################################################################################
## HELPER METHODS
################################################################################


def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Compresses a given Keras model using the provided method and '
            'parameterization of the method.'
        ),
    )
    parser.add_argument(
        'model',
        help="path to a valid Keras h5 file containing a trained graph.",
        metavar="model_path.h5",
        type=str,
    )
    parser.add_argument(
        'dataset_name',
        help="name of the dataset to be used for training.",
        metavar="dataset_name",
        type=str,
    )
    parser.add_argument(
        'dataset_file',
        help="comma-separated-valued file containing our training data.",
        metavar="data.cvs",
        type=str,
    )
    parser.add_argument(
        '--compression_method',
        '-c',
        default='weight-magnitude',
        help=(
            "name of the extraction algorithm to be used to generate our "
            "rule set."
        ),
        metavar="name",
        choices=[
            'unit-magnitude',
            'weight-magnitude',
            'brain-damage',
            'brain-surgeon',
        ],
    )
    parser.add_argument(
        '--output_file',
        '-o',
        default="compressed_model.h5",
        help=(
            "output filename to be used to serialize the compressed output "
            "model. If not given, then it will serialize the output to "
            "compressed_model.h5 by default."
        ),
        metavar="compressed_model.h5",
        type=str,
    )
    parser.add_argument(
        '--compress_rate',
        '-r',
        default=0.9,
        type=float,
        help=(
            "Number between 0 and 1 indicating the final compression rate "
            "for the input model after applying the requested method. If not"
            "given, then we will use a compression rate of 90% by default."
        ),
        metavar="fraction",

    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help=(
            "Batch size to be used for retraining purposes coming after "
            "pruning. By default we will use a batch size of 32."
        ),
        metavar="batch",
    )
    parser.add_argument(
        "--pruning_epochs",
        default=10,
        type=int,
        help=(
            "Number of retraining epochs to be used after pruning a model. By "
            "default we will use 10 epochs."
        ),
        metavar="epochs",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        '--test_percent',
        '-t',
        default=0.2,
        type=float,
        help=(
            "Number between 0 and 1 indicating which fraction of the total "
            "provided data should be taken to be the test dataset. If 0, then "
            "the entire dataset will be used for training."
        ),
        metavar="fraction",

    )
    return parser


def get_majority_class(y):
    (unique, counts) = np.unique(y, return_counts=True)
    freqs = sorted(
        list(zip(unique, counts)),
        key=lambda x: -x[1],
    )
    maj_class, _ = freqs[0]
    maj_acc = accuracy_score(
        y,
        [maj_class for _ in y],
    )
    return maj_class, maj_acc


def _evalutate_model(
    model,
    dataset_descr,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
):
    data_table = PrettyTable()
    data_table.field_names = [
        "Dataset Name",
        "# of Samples",
        "# of Features",
        "# of Classes",
        "Majority Class",
        "Majority Class Accuracy",
        "Model Accuracy",
    ]
    out_train_vals = model.predict(
        X_train,
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
            else 0
        ),
    )
    if isinstance(out_train_vals, list):
        raise ValueError(out_train_vals)
    model_train_acc = accuracy_score(
        y_train,
        np.argmax(out_train_vals, axis=-1),
    )

    train_maj_class, train_maj_acc = get_majority_class(y_train)
    data_table.add_row([
        f'{dataset_descr.name} (train)',  # Name
        len(X_train),  # Output samples
        len(dataset_descr.feature_names),  # Output features
        len(dataset_descr.output_classes),  # Output classes
        dataset_descr.output_classes[train_maj_class].name,  # Majority class
        round(train_maj_acc, 4),  # Majority class acc
        round(model_train_acc, 4),  # Rule extraction accuracy
    ])

    if X_test is not None:
        out_test_vals = model.predict(
            X_test,
            verbose=(
                1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                else 0
            ),
        )
        model_test_acc = accuracy_score(
            y_test,
            np.argmax(out_test_vals, axis=-1),
        )

        test_maj_class, test_maj_acc = get_majority_class(y_test)
        data_table.add_row([
            f'{dataset_descr.name} (test)',  # Name
            len(X_test),  # Output samples
            len(dataset_descr.feature_names),  # Output features
            len(dataset_descr.output_classes),  # Output classes
            dataset_descr.output_classes[test_maj_class].name,  # Majority class
            round(test_maj_acc, 4),  # Majority class acc
            round(model_test_acc, 4),  # Rule extraction accuracy
        ])

    # Finally print out the results
    print()
    print(data_table)
    print()


################################################################################
## MAIN METHOD
################################################################################

def main():
    """
    Our main entry point method for our program's execution. Instantiates
    the argument parser and calls the appropriate methods using the given
    flags.
    """

    # First generate our argument parser
    parser = build_parser()
    args = parser.parse_args()

    # Now set up our logger
    logging.getLogger().setLevel(
        logging.DEBUG if args.debug else logging.INFO
    )
    logging.basicConfig(
        format='[%(levelname)s] %(message)s'
    )

    # Time to get our dataset descriptor
    dataset_descr = dataset_configs.get_data_configuration(
        dataset_name=args.dataset_name,
    )

    # Read the input data
    X, y, data = dataset_descr.read_data(args.dataset_file)

    # Obtain our test and train datasets
    if args.test_percent:
        split_gen = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.test_percent,
            random_state=42,
        )
        [(train_indices, test_indices)] = split_gen.split(X, y)
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
    else:
        # Else there is no testing data to be used
        X_train, y_train = X, y
        X_test, y_test = None, None

    # Preprocess data if needed
    if dataset_descr.preprocessing is not None:
        if X_test is not None:
            X_train, y_train, X_test, y_test = dataset_descr.preprocessing(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        else:
            X_train, y_train = dataset_descr.preprocessing(
                X_train=X_train,
                y_train=y_train,
            )

    # Deserialize the given Keras model
    nn_model = load_model(args.model)
    nn_model.summary()

    print("*" * 35, "ORIGINAL MODEL", "*" * 35)
    _evalutate_model(
        model=nn_model,
        dataset_descr=dataset_descr,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    compression_name = args.compression_method.lower()
    if compression_name == "weight-magnitude":
        compressed_model = weight_magnitude_compress(
            nn_model,
            X_train=X_train,
            y_train=tf.keras.utils.to_categorical(y_train),
            batch_size=args.batch_size,
            pruning_epochs=args.pruning_epochs,
            target_sparsity=args.compress_rate,
            initial_sparsity=0.1,
            ignore_layer_fn=lambda l: "output_dense" in l.name,
        )
    elif compression_name == "unit-magnitude":
        compressed_model = neuron_magnitude_compress(
            nn_model,
            X_train=X_train,
            y_train=tf.keras.utils.to_categorical(y_train),
            batch_size=args.batch_size,
            pruning_epochs=args.pruning_epochs,
            target_sparsity=args.compress_rate,
            initial_sparsity=0.1,
            ignore_layer_fn=lambda l: "output_dense" in l.name,
        )
    elif compression_name == "brain-damage":
        raise NotImplementedError("Brain-Damage")

    elif compression_name == "brain-surgeon":
        raise NotImplementedError("Brain-Surgeon")
    else:
        raise ValueError(f"Unrecognized compression method {compression_name}")

    print("*" * 35, "COMPRESSED MODEL", "*" * 35)
    compressed_model.summary()
    _evalutate_model(
        model=compressed_model,
        dataset_descr=dataset_descr,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    for i, layer in enumerate(compressed_model.layers):
        print("Layer at depth is", i + 1, "is", layer)
        print("\tLayer has input shape", layer.input_shape)
        print("\tLayer has output shape", layer.output_shape)
        print("\tLayer has trainable weights", list(map(lambda x: x.name, layer.trainable_weights)))
        print("\tLayer has trainable weight shapes", list(map(lambda x: x.shape, layer.trainable_weights)))
        if isinstance(layer, tf.keras.layers.Dense):
            # Then let's try and see how can we prune this entire model
            kernel = layer.kernel.numpy()
            col_sums = np.sum(np.abs(kernel), axis=0)
            col_sums = np.isclose(col_sums, 0)
            print("Can prune", np.sum(col_sums), "out of", kernel.shape[1], "neurons")

    # And serialize it
    compressed_model.save(args.output_file)

################################################################################
## ENTRY POINT
################################################################################

if __name__ == '__main__':
    os.environ['PYTHONHASHSEED'] = str(_RANDOM_SEED)
    tf.random.set_seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)
    random.seed(_RANDOM_SEED)
    prev_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            # execute code that will generate warnings
            sys.exit(main())
    finally:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = prev_tf_log_level

    # If we reached this point, then we are exiting with an error code
    sys.exit(1)
