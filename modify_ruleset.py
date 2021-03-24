#!/usr/bin/env python3

"""
Script to modify a given ruleset by deleting some rules or changing the scoring
mechanism used by the ruleset.
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
import argparse
import logging
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import warnings
from prettytable import PrettyTable


from dnn_rem.evaluate_rules.evaluate import evaluate as ruleset_evaluate
from dnn_rem.experiment_runners import dataset_configs
from dnn_rem.rules.ruleset import RuleScoreMechanism
from dnn_rem.utils.resources import resource_compute


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
            'Modifies a given ruleset and evaluates in the provided dataset.'
        ),
    )
    parser.add_argument(
        'ruleset',
        help="path to a valid Keras h5 file containing a trained graph.",
        metavar="rulset_file.rules",
        type=str,
    )
    parser.add_argument(
        '--dataset_name',
        help="name of the dataset to be used for training.",
        metavar="dataset_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--dataset_file',
        help="comma-separated-valued file containing our training data.",
        metavar="data.cvs",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--output_file',
        '-o',
        default=None,
        help=(
            "output filename where we will dump our experiment's results. If "
            "not given, then we will use the same name as the input file but "
            "attach the string '_modified' to the end of the name."
        ),
        metavar="output_filename.rules",
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
        default=None,
        help=(
            "Mechanism to be used for finding a given class' score. By default "
            "it will preserve whatever mechanism is being used by the provided "
            "ruleset."
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
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
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


def _evalutate_ruleset(
    ruleset,
    dataset_descr,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    dataset_name="",
):
    data_table = PrettyTable()
    data_table.field_names = [
        "Dataset Name",
        "# of Samples",
        "# of Features",
        "# of Classes",
        "Majority Class",
        "Majority Class Accuracy",
        f"Ruleset Accuracy",
        "Ruleset Size",
        "Average Rule Length",
    ]
    re_train_results = ruleset_evaluate(
        ruleset=ruleset,
        X_test=X_train,
        y_test=y_train,
    )

    avg_rule_length = np.array(re_train_results['av_n_terms_per_rule'])
    avg_rule_length *= np.array(re_train_results['n_rules_per_class'])
    avg_rule_length = sum(avg_rule_length)
    avg_rule_length /= sum(re_train_results['n_rules_per_class'])
    train_maj_class, train_maj_acc = get_majority_class(y_train)
    data_table.add_row([
        f'{dataset_name} (train)',  # Name
        len(X_train),  # Output samples
        len(dataset_descr.feature_names),  # Output features
        len(dataset_descr.output_classes),  # Output classes
        dataset_descr.output_classes[train_maj_class].name,  # Majority class
        round(train_maj_acc, 4),  # Majority class acc
        round(re_train_results['acc'], 4),  # Rule extraction accuracy
        sum(re_train_results['n_rules_per_class']),  # Rule extracted size
        round(avg_rule_length, 4),  # Rule extracted average length
    ])

    if X_test is not None:
        re_test_results = ruleset_evaluate(
            ruleset=ruleset,
            X_test=X_test,
            y_test=y_test,
        )
        test_maj_class, test_maj_acc = get_majority_class(y_test)
        data_table.add_row([
            f'{dataset_name} (test)',  # Name
            len(X_test),  # Output samples
            len(dataset_descr.feature_names),  # Output features
            len(dataset_descr.output_classes),  # Output classes
            dataset_descr.output_classes[test_maj_class].name,  # Majority class
            round(test_maj_acc, 4),  # Majority class acc
            round(re_test_results['acc'], 4),  # Rule extraction accuracy
            "N/A",  # Rule extracted size
            "N/A",  # Rule extracted average length
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

    # Now deserialize the provided ruleset
    with open(args.ruleset, 'rb') as f:
        ruleset = pickle.load(f)

    # Time to get our dataset descriptor
    X_train, y_train = None, None
    dataset_descr = None
    if args.dataset_name or args.dataset_file:
        if not args.dataset_file:
            raise ValueError(
                'If dataset name is provided then a path to a file should also '
                'be provided via the --dataset_file flag.'
            )
        if not args.dataset_name:
            raise ValueError(
                'If dataset file is provided then the name of the used dataset '
                'should also be provided via the --dataset_name flag.'
            )
        dataset_descr = dataset_configs.get_data_configuration(
            dataset_name=args.dataset_name,
        )
        X, y, data = dataset_descr.read_data(args.dataset_file)

        # Obtain our test and train datasets
        if args.test_percent:
            split_gen = ShuffleSplit(
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

    # If data was provided, let's evaluate the ruleset in the given dataset
    if X_train is not None:
        print("*" * 35, "ORIGINAL RULESET", "*" * 35)
        _evalutate_ruleset(
            ruleset,
            dataset_descr=dataset_descr,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=args.dataset_name,
        )

    # Get the rule scoring mechanism
    if args.rule_score_mechanism is not None:
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
    output_file = args.output_file
    if output_file is None:
        # Then we will use the original filename to construct the resulting
        # output file
        path, extension = os.path.splitext(args.ruleset)
        output_file = path + "_modified" + extension
    with open(output_file, 'wb') as f:
        pickle.dump(ruleset, f)

    # And finalize by evaluating the modified ruleset!
    if X_train is not None:
        print("*" * 35, "MODIFIED RULESET", "*" * 35)
        _evalutate_ruleset(
            ruleset,
            dataset_descr=dataset_descr,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=args.dataset_name,
        )


################################################################################
## ENTRY POINT
################################################################################

if __name__ == '__main__':
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
