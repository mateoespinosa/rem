#!/usr/bin/env python3

"""
Executable tool for extracting a ruleset from a TF model serialized as a .h5
file.
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

from dnn_rem.evaluate_rules.evaluate import evaluate as ruleset_evaluate
from dnn_rem.experiment_runners import dataset_configs
from dnn_rem.experiment_runners.manager import RuleExMode
from dnn_rem.extract_rules.pedagogical import extract_rules as pedagogical
from dnn_rem.extract_rules.rem_d import extract_rules as rem_d
from dnn_rem.extract_rules.rem_t import extract_rules as rem_t
from dnn_rem.model_training.build_and_train_model import load_model
from dnn_rem.rules.ruleset import RuleScoreMechanism
from dnn_rem.utils.resources import resource_compute

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
            'Extracts a ruleset from given h5 trained keras model.'
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
        '--rule_extractor',
        default='rem-d',
        help=(
            "name of the extraction algorithm to be used to generate our "
            "rule set."
        ),
        metavar="name",
        choices=['rem-d', 'pedagogical', 'rem-t'],
    )
    parser.add_argument(
        '--output_file',
        '-o',
        default="out_rules.rules",
        help=(
            "output filename where we will dump our experiment's results. If "
            "not given, then we will use out_rules.rules"
        ),
        metavar="output_filename.rules",
        type=str,
    )
    parser.add_argument(
        '--last_activation',
        default=None,
        help=(
            "output activation to be applied to elements of last layer in case "
            "it was merged into the subsequent loss. It must be a valid "
            "string name to a Keras activation."
        ),
        metavar="act_name",
        type=str,
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
    parser.add_argument(
        '--rule_score_mechanism',
        default='majority',
        help=(
            "Mechanism to be used for finding a given class' score. By default "
            "we use a majority class scoring."
        ),
        metavar="name",
        choices=['majority', 'accuracy', 'hillclimb', 'confidence'],

    )
    parser.add_argument(
        '--rule_drop_percent',
        default=0,
        type=float,
        help=(
            "Fraction of lowest scoring rules to drop after rules have been "
            "extracted. By default we drop no rules."
        ),
        metavar="percent",
    )

    parser.add_argument(
        '--num_workers',
        default=6,
        type=int,
        help=(
            "Number of workers to be used for parallel rule extraction when "
            "this is allowed."
        ),
        metavar="n",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    return parser


def get_rule_extractor(
    extractor_name,
    feature_names=None,
    output_class_names=None,
    **extractor_params,
):
    name = extractor_name.lower()
    if name == "rem-d":
        def _run(*args, **kwargs):
            return rem_d(
                *args,
                **kwargs,
                **extractor_params,
                feature_names=feature_names,
                output_class_names=output_class_names,
            )
        return RuleExMode(
            mode='REM-D',
            run=_run,
        )

    if name == "pedagogical":
        def _run(*args, **kwargs):
            kwargs.pop("train_labels", None)
            return pedagogical(
                *args,
                **kwargs,
                **extractor_params,
                feature_names=feature_names,
                output_class_names=output_class_names,
            )
        return RuleExMode(
            mode='pedagogical',
            run=_run,
        )

    if name == "rem-t":
        return RuleExMode(
            mode='REM-T',
            run=lambda *args, **kwargs: rem_t(
                *args,
                **kwargs,
                **extractor_params,
                feature_names=feature_names,
                output_class_names=output_class_names,
            ),
        )

    raise ValueError(
        f'Given rule extractor "{extractor_name}" is not a valid rule '
        f'extracting algorithm. Valid modes are "REM-D" or '
        f'"pedagogical".'
    )


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

    # Get the rule scoring mechanism
    rule_score_mechanism = None
    mechanism_name = args.rule_score_mechanism
    search_name = mechanism_name.lower()
    for enum_entry in RuleScoreMechanism:
        if enum_entry.name.lower() == search_name:
            rule_score_mechanism = enum_entry
    if rule_score_mechanism is None:
        raise ValueError(
            f'We do not support score mode "{mechanism_name}" as a rule '
            f'scoring mechanism. We support the following rule scoring '
            f'mechanisms: {list(map(lambda x: x.name, RuleScoreMechanism))}'
        )

    # Get the rule extractor
    extractor_params = {}
    rule_extractor_name = args.rule_extractor.lower()
    if rule_extractor_name == "rem-d":
        extractor_params["num_workers"] = args.num_workers
        extractor_params["last_activation"] = args.last_activation
    rule_extractor = get_rule_extractor(
        extractor_name=rule_extractor_name,
        feature_names=dataset_descr.feature_names,
        output_class_names=list(map(
            lambda x: x.name,
            dataset_descr.output_classes,
        )),
        **extractor_params,
    )

    # We will summarize our used data in a table for analysis purposes
    data_table = PrettyTable()
    data_table.field_names = [
        "Dataset Name",
        "# of Samples",
        "# of Features",
        "# of Classes",
        "Majority Class",
        "Majority Class Accuracy",
        "NN Accuracy",
        f"{rule_extractor_name} Accuracy",
        f"{rule_extractor_name} Fidelity",
        "Extraction Time (sec)",
        "Extraction Memory (MB)",
        "Ruleset Size",
        "Average Rule Length",
    ]

    # Read the input data
    X, y, data = dataset_descr.read_data(args.dataset_file)

    # Obtain our test and train datasets
    if args.test_percent:
        split_gen = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.test_percent,
            random_state=_RANDOM_SEED,
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

    train_maj_class, train_maj_acc = get_majority_class(y_train)

    # Deserialize the given Keras model
    nn_model = load_model(args.model)
    nn_model.summary()

    # Evaluate the model in our training and test data for reporting purposes
    out_train_vals = nn_model.predict(
        X_train,
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
            else 0
        ),
    )
    nn_train_accuracy = accuracy_score(
        y_train,
        np.argmax(out_train_vals, axis=-1),
    )

    if X_test is not None:
        out_test_vals = nn_model.predict(
            X_test,
            verbose=(
                1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                else 0
            ),
        )
        nn_test_accuracy = accuracy_score(
            y_test,
            np.argmax(out_test_vals, axis=-1),
        )

    # Now we have everything we need to call our rule-extractor
    ruleset, re_train_time, re_train_memory = resource_compute(
        function=rule_extractor.run,
        model=nn_model,
        train_data=X_train,
        train_labels=y_train,
    )

    # Now let's assign scores to our rules depending on what scoring
    # function we were asked to use for this experiment
    ruleset.rank_rules(
        X=X_train,
        y=y_train,
        score_mechanism=rule_score_mechanism,
    )

    # Drop any rules if we are interested in dropping them
    ruleset.eliminate_rules(args.rule_drop_percent)

    # Serialize the generated rules
    with open(args.output_file, 'wb') as f:
        pickle.dump(ruleset, f)

    # And finalize by evaluating the extracted ruleset!
    re_train_results = ruleset_evaluate(
        ruleset=ruleset,
        X_test=X_train,
        y_test=y_train,
        high_fidelity_predictions=np.argmax(
            nn_model.predict(X_train),
            axis=1
        ),
    )

    avg_rule_length = np.array(re_train_results['av_n_terms_per_rule'])
    avg_rule_length *= np.array(re_train_results['n_rules_per_class'])
    avg_rule_length = sum(avg_rule_length)
    avg_rule_length /= sum(re_train_results['n_rules_per_class'])

    data_table.add_row([
        f'{args.dataset_name} (train)',  # Name
        len(X_train),  # Output samples
        len(dataset_descr.feature_names),  # Output features
        len(dataset_descr.output_classes),  # Output classes
        dataset_descr.output_classes[train_maj_class].name,  # Majority class
        round(train_maj_acc, 4),  # Majority class acc
        round(nn_train_accuracy, 4),  # NN Accuracy
        round(re_train_results['acc'], 4),  # Rule extraction accuracy
        round(re_train_results['fid'], 4),  # Rule extraction fidelity
        round(re_train_time,  4),  # Rule extraction time taken
        round(re_train_memory, 4),  # Rule extraction memory taken
        sum(re_train_results['n_rules_per_class']),  # Rule extracted size
        round(avg_rule_length, 4),  # Rule extracted average length
    ])

    if X_test is not None:
        re_test_results = ruleset_evaluate(
            ruleset=ruleset,
            X_test=X_test,
            y_test=y_test,
            high_fidelity_predictions=np.argmax(
                nn_model.predict(X_test),
                axis=1
            ),
        )
        test_maj_class, test_maj_acc = get_majority_class(y_test)
        data_table.add_row([
            f'{args.dataset_name} (test)',  # Name
            len(X_test),  # Output samples
            len(dataset_descr.feature_names),  # Output features
            len(dataset_descr.output_classes),  # Output classes
            dataset_descr.output_classes[test_maj_class].name,  # Majority class
            round(test_maj_acc, 4),  # Majority class acc
            round(nn_test_accuracy, 4),  # NN Accuracy
            round(re_test_results['acc'], 4),  # Rule extraction accuracy
            round(re_test_results['fid'], 4),  # Rule extraction fidelity
            "N/A",  # Rule extraction time taken
            "N/A",  # Rule extraction memory taken
            "N/A",  # Rule extracted size
            "N/A",  # Rule extracted average length
        ])

    # Finally print out the results
    print()
    print(data_table)
    print()


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
