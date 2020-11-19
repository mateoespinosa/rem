#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import warnings
import yaml


from dnn_rem.model_training import generate_data
from dnn_rem.model_training.split_data import load_data
from dnn_rem.experiment_runners.cross_validation import cross_validate_re
from dnn_rem.experiment_runners import dnn_re
from dnn_rem.experiment_runners.manager import ExperimentManager


################################################################################
## HELPER METHODS
################################################################################


def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description='Process some integers.'
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help="initial configuration YAML file for our experiment's setup.",
        metavar="file.yaml",
    )
    parser.add_argument(
        '--folds',
        '-f',
        default=None,
        help='how many folds to use for our data partitioning.',
        metavar='N',
        type=int,
    )
    parser.add_argument(
        '--initialisation_trials',
        '-i',
        default=None,
        help=(
            "how many initialisations for our model's initial parameters do we "
            "want to try before running our training. If less than or equal to "
            "one, then we will simply use a random initialisation."
        ),
        metavar="N",
        type=int,
    )
    parser.add_argument(
        '--dataset_name',
        default=None,
        help="name of the dataset to be used for training.",
        metavar="name",
    )
    parser.add_argument(
        '--dataset_file',
        default=None,
        help="comma-separated-valued file containing our training data.",
        metavar="data.cvs",
    )
    parser.add_argument(
        '--rule_extractor',
        default=None,
        help=(
            "name of the extraction algorithm to be used to generate our "
            "rule set."
        ),
        metavar="name",
        choices=['DeepRED_C5', 'Pedagogical'],
    )
    parser.add_argument(
        '--grid_search',
        action="store_true",
        default=False,
        help=(
            "whether we want to do a grid search over our model's "
            "hyperparameters. If the results of a previous grid search are "
            "found in the provided output directory, then we will use those "
            "rather than starting a grid search from scratch. This means that "
            "turning this flag on will overwrite any hyperparameters you "
            "provide as part of your configuration (if any)."
        ),
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use the same directory as our dataset."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--tmp_dir',
        default=None,
        help=(
            "temporary directory to use for scratch work. If not provided, "
            "then we will instantiate our own temporary directory to be "
            "removed at the end of the program's execution."
        ),
        metavar="path",

    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    return parser


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

    # Now see if a config file was passed or if all arguments come from the
    # command line
    if args.config is not None:
        # First deserialize and validate the given config file if it was
        # given
        if not os.path.exists(args.config):
            raise ValueError(
                f'Given config file "{args.config}" is not a valid path.'
            )
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}
        if args.folds is None:
            # Then default it to 1
            # Here and below: the reason why we do not default it in the
            # argparser itself is because we want to have the option to
            # run between the mode where a config file has been provided but we
            # want to overwrite certain attributes of it and the case where no
            # config file was provided so we expect all arguments coming from
            # the command line.
            args.folds = 1
        if args.rule_extractor is None:
            # Default it to use our rule generation algorithm
            args.rule_extractor = "DeepRED_C5"
        if args.initialisation_trials is None:
            # Then default it to not making any initalisation
            args.initialisation_trials = 1
        if None in [
            args.dataset_name,
            args.dataset_file,
        ]:
            # Then time to complain
            raise ValueError(
                'We expect to either be provided a valid configuration '
                'YAML file or to be explicitly provided arguments '
                'dataset_name and dataset_file. Otherwise, we do not have a '
                'complete parameterization for our experiment.'
            )

    # And we overwrite any arguments that were provided outside of our
    # config file:
    if args.folds is not None:
        config["n_folds"] = args.folds
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    if args.rule_extractor is not None:
        config["rule_extractor"] = args.rule_extractor
    if args.dataset_file is not None:
        config["dataset_file"] = args.dataset_file
    if args.initialisation_trials is not None:
        config["initialisation_trials"] = args.initialisation_trials
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.tmp_dir is not None:
        config["tmp_dir"] = args.tmp_dir

    # Time to initialize our experiment manager
    manager = ExperimentManager(config)
    if args.tmp_dir is None:
        # Then report which directory we will be using during debug
        logging.debug(
            f"Using {manager.TEMP_DIR} as our temporary directory..."
        )

    # use that to open up our dataset
    X, y = load_data(manager.DATASET_INFO, manager.DATA_FP)
    # And do any required preprocessing
    if manager.DATASET_INFO.preprocessing:
        X, y = manager.DATASET_INFO.preprocessing(X, y)

    # Generate our neural network, train it, and then extract the ruleset that
    # approximates it from it
    generate_data.run(
        X=X,
        y=y,
        manager=manager,
        split_data=True,
        use_grid_search=args.grid_search,
        find_best_initialisation=(manager.INITIALISATION_TRIALS > 1),
        generate_fold_data=True,
    )

    # Perform n fold cross validated rule extraction on the dataset
    cross_validate_re(
        X=X,
        y=y,
        manager=manager,
    )

    # And that's all folks
    return 0

################################################################################
## ENTRY POINT
################################################################################

if __name__ == '__main__':
    # Make sure we do not print any useless warnings in our script to reduce
    # the amount of noise
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
