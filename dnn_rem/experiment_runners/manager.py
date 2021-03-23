"""
Class for managing directory hierarchies when running experiments with the
rule generation algorithm.
"""

from collections import namedtuple
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

import logging
import numpy as np
import os
import pathlib
import random
import shutil
import tempfile
import tensorflow as tf
import time
import yaml

from . import dataset_configs
from dnn_rem.extract_rules.pedagogical import extract_rules as pedagogical
from dnn_rem.extract_rules.rem_t import extract_rules as rem_t
from dnn_rem.extract_rules.rem_d import extract_rules as rem_d
from dnn_rem.rules.ruleset import RuleScoreMechanism


# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', ['mode', 'run'])

# The different stages we will have in our experimentation
EXPERIMENT_STAGES = [
    "data_split",
    "fold_split",
    "grid_search",
    "initialisation_trials",
    "nn_train",
    "rule_extraction",
]

################################################################################
## INPUT DATA HELPERS
################################################################################


def split_serializer(splits, file_path):
    # Helper method to serialize data split indices into a file
    with open(file_path, 'w') as f:
        for (train_indices, test_indices) in splits:
            f.write('train ' + ' '.join([str(i) for i in train_indices]) + '\n')
            f.write('test ' + ' '.join([str(i) for i in test_indices]) + '\n')
    return splits


def split_deserializer(path):
    # Helper method to deserialize data split indices from a given file.
    result = []
    with open(path, 'r') as file:
        lines = file.readlines()
        assert len(lines) % 2 == 0, (
           f"Expected even number of lines in file {path} but got {len(lines)} "
           "instead."
        )
        for i in range(len(lines) // 2):
            result.append((
                list(map(int, lines[(i * 2)].split(' ')[1:])),
                list(map(int, lines[(i * 2) + 1].split(' ')[1:]))
            ))
    return result


def stratified_k_fold_split(
    X,
    y,
    n_folds=1,
    test_size=0.2,
    random_state=None,
):
    # Helper method to return a list with n_folds tuples (X_train, y_train)
    # after partitioning the given X, y dataset using stratified splits.
    result = []
    if n_folds == 1:
        # Degenerate case: let's just dump all our indices as our single fold
        split_gen = ShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        # Split data
        split_gen = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )

    # Save indices
    for train_indices, test_indices in split_gen.split(X, y):
        result.append((train_indices, test_indices))

    return result


################################################################################
## EXPERIMENT MANAGER MAIN CLASS
################################################################################

class ExperimentManager(object):
    """
    Manager class for organizing and keeping track of the directory
    structure we will use during experimentation.
    Will contain several fields related to directories and files that we
    will use to keep track of results, data, and intermediate temporary files
    as we work through our experiments.

    Our results directory will follow the following structure:
        <dataset-name>
            data.csv

        <results-directory>
            cross_validation/
                <n>_folds/
                    rule_extraction/
                        <rule ex mode>/
                            results.csv
                            rules_extracted/
                                fold_<n>.rules
                    trained_models/
                        fold_<n>_model.h5
                    data_split_indices.txt
            neural_network_initialisation/
                re_results.csv
                grid_search_results.txt
                data_split_indices.txt
                best_initialisation.h5
    """

    def __init__(self, config, start_rerun_stage=None):

        # Let's see if we allow for checkpointing or if we want to always do
        # a rerun
        self._forced_rerun = start_rerun_stage in ["all", EXPERIMENT_STAGES[0]]
        self._start_rerun_stage = start_rerun_stage

        # Some hidden state for management purposes only
        self._start_time = time.time()  # For timing purposes
        self._previous_seed = os.environ.get('PYTHONHASHSEED')

        # Validate our provided config
        self.validate_config_object(config)

        # Obtain our rule score mode
        self.RULE_SCORE_MECHANISM = None
        mechanism_name = config.get("rule_score_mechanism", "majority")
        search_name = mechanism_name.lower()
        for enum_entry in RuleScoreMechanism:
            if enum_entry.name.lower() == search_name:
                self.RULE_SCORE_MECHANISM = enum_entry
        if self.RULE_SCORE_MECHANISM is None:
            raise ValueError(
                f'We do not support score mode "{mechanism_name}" as a rule '
                f'scoring mechanism. We support the following rule scoring '
                f'mechanisms: {list(map(lambda x: x.name, RuleScoreMechanism))}'
            )
        self.RULE_DROP_PRECENT = config.get("rule_elimination_percent", 0)

        # Now time to build all the directory variables we will need to
        # construct our experiment
        self.DATASET_INFO = dataset_configs.get_data_configuration(
            dataset_name=config["dataset_name"],
        )

        self.DATA_FP = config["dataset_file"]
        # How many trials we will attempt for finding our best initialisation
        self.INITIALISATION_TRIALS = config["initialisation_trials"]
        self.N_FOLDS = config["n_folds"]
        self.HYPERPARAMS = config["hyperparameters"]

        # What percent of our data will be used as test data
        self.PERCENT_TEST_DATA = config.get("percent_test_data", 0.2)

        # The number of decimals used to report floating point numbers
        self.ROUNDING_DECIMALS = config.get("rounding_decimals", 3)

        # And build our rule extractor
        self.RULE_EXTRACTOR = self.get_rule_extractor(
            config.get("rule_extractor", "rem_d"),
            **config.get("extractor_params", {})
        )

        # A random seed to use for deterministic training
        self.RANDOM_SEED = config.get("random_seed", time.time())

        # Where all our results will be dumped. If not provided as part of the
        # experiment's config, then we will use the same parent directory as the
        # datafile we are using
        self.experiment_dir = config.get(
            "output_dir",
            pathlib.Path(self.DATA_FP).parent
        )

        # Time to set up our grid-search config
        self.GRID_SEARCH_PARAMS = config.get(
            "grid_search_params",
            {},
        )

        # And our initialization trial
        self.BEST_INIT_METRIC_NAME = config.get(
            "initialisation_trial_metric",
            "acc",
        )

        # <dataset_name>/cross_validation/<n>_folds/
        cross_val_dir = os.path.join(self.experiment_dir, 'cross_validation')
        self.SUMMARY_FILE = os.path.join(cross_val_dir, "summary.txt")
        self.N_FOLD_CV_DP = os.path.join(
            cross_val_dir,
            f'{self.N_FOLDS}_folds'
        )
        self.N_FOLD_CV_SPLIT_INDICES_FP = os.path.join(
            self.N_FOLD_CV_DP,
            'data_split_indices.txt'
        )

        # <dataset_name>/cross_validation/<n>_folds/rule_extraction/<rule_ex_mode>/rules_extracted/
        self.N_FOLD_RULE_EX_MODE_DP = os.path.join(
            self.N_FOLD_CV_DP,
            'rule_extraction',
            self.RULE_EXTRACTOR.mode
        )
        self.N_FOLD_RESULTS_FP = os.path.join(
            self.N_FOLD_RULE_EX_MODE_DP,
            'results.csv',
        )
        self.N_FOLD_RULES_DP = os.path.join(
            self.N_FOLD_RULE_EX_MODE_DP,
            'rules_extracted',
        )
        self.n_fold_rules_fp = lambda fold: os.path.join(
            self.N_FOLD_RULES_DP,
            f'fold_{fold}.rules',
        )
        self.rules_fp = os.path.join(self.N_FOLD_RULES_DP, 'fold.rules')

        # <dataset_name>/cross_validation/<n>_folds/trained_models/
        self.N_FOLD_MODELS_DP = os.path.join(
            self.N_FOLD_CV_DP,
            'trained_models',
        )
        self.n_fold_model_fp = (
            lambda fold: os.path.join(
                self.N_FOLD_MODELS_DP,
                f'fold_{fold}_model.h5'
            )
        )
        model_fp = os.path.join(self.N_FOLD_MODELS_DP, 'model.h5')

        # <dataset_name>/neural_network_initialisation/
        nn_init_dir = os.path.join(
            self.experiment_dir,
            'neural_network_initialisation'
        )
        self.NN_INIT_GRID_RESULTS_FP = os.path.join(
            nn_init_dir,
            'grid_search_results.txt'
        )
        self.NN_INIT_SPLIT_INDICES_FP = os.path.join(
            nn_init_dir,
            'data_complete_split_indices.txt'
        )
        self.NN_INIT_RE_RESULTS_FP = os.path.join(
            nn_init_dir,
            're_results.csv'
        )
        self.BEST_NN_INIT_FP = os.path.join(
            nn_init_dir,
            'best_initialisation.h5'
        )

        # And time for some data and directory initialization!
        if self._start_rerun_stage:
            logging.warning(
                "We will overwrite every previous result in "
                f'"{self.experiment_dir}" starting from stage '
                f'"{self._start_rerun_stage}"'
            )

        # Set up all the directories we will need
        self._initialize_directories()

        # For reference purposes, let's dump our effective config file into
        # the experiment directory to make sure we can always reproduce the
        # results generated here
        with open(os.path.join(self.experiment_dir, "config.yaml"), 'w') as f:
            f.write(yaml.dump(config, sort_keys=True))

        # And initialize our data splits
        self._initialize_data()

    def __enter__(self):
        """
        Enter code. Setup any initial state that will be required for the
        experiment to run as we want it to.
        """
        self._start_time = time.time()
        # Save the previous seed in case we need to reset it
        self._previous_seed = os.environ.get('PYTHONHASHSEED')
        if self.RANDOM_SEED:
            os.environ['PYTHONHASHSEED'] = str(self.RANDOM_SEED)
            tf.random.set_seed(self.RANDOM_SEED)
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Safe exit code from our experiment execution. Make sure
        we remove our temporary folders and/or state we changed as soon as we
        are out.
        """
        if self.RANDOM_SEED and self._previous_seed:
            # Then reset our environment
            os.environ['PYTHONHASHSEED'] = self._previous_seed

        print(
            "~" * 20,
            "Experiment successfully terminated after",
            round(time.time() - self._start_time, 3),
            "seconds",
            "~" * 20,
        )

    @staticmethod
    def validate_config_object(config):
        """
        Validates the given 'config' deserialized as a given YAML object.
        Makes sure all required fields are provided and that they have
        sensible values. If this is not the case, then a ValueError will be
        thrown.
f
        If this grows too much, I strongly suggest moving to (a) either move
        to protobuf or (b) use external packages to perform schema validation.

        :param config:  The configuration file deserialized as a YAML object.
        """
        for field_name in [
            "dataset_file",
            "dataset_name",
            "n_folds",
            "hyperparameters",
        ]:
            if field_name not in config:
                raise ValueError(
                    f'Expected field "{field_name}" to be provided as part of '
                    f"the experiment's config."
                )

        # Time to check the given dataset is a valid data set
        if not dataset_configs.is_valid_dataset(config["dataset_name"]):
            raise ValueError(
                f'Given dataset name "{config["dataset_name"]} is not a '
                f'supported dataset. We currently support the following '
                f'datasets: {dataset_configs.AVAILABLE_DATASETS}.'
            )

        # And also check our model hyperparameters
        for hyper_field in [
            "batch_size",
            "epochs",
            "layer_units",
        ]:
            if hyper_field not in config["hyperparameters"]:
                raise ValueError(
                    f'Expected hyper-parameters "{hyper_field}" to be provided '
                    f"as part of the experiment's config's hyperparameters "
                    f"field."
                )

    def get_rule_extractor(self, extractor_name, **extractor_params):
        name = extractor_name.lower()
        if name == "rem-d":
            loss_function = self.HYPERPARAMS.get(
                "loss_function",
                "softmax_xentr",
            )
            last_activation = self.HYPERPARAMS.get(
                "last_activation",
                "softmax",
            )
            # We set the last activation to None here if it is going to be
            # be included in the network itself. Otherwise, we request our
            # rule extractor to explicitly perform the activation on the last
            # layer as this was merged into the loss function
            last_activation = (
                last_activation if last_activation in loss_function else None
            )

            def _run(*args, **kwargs):
                kwargs.pop("train_labels", None)
                return rem_d(
                    *args,
                    **kwargs,
                    **extractor_params,
                    last_activation=last_activation,
                    feature_names=self.DATASET_INFO.feature_names,
                    output_class_names=list(map(
                        lambda x: x.name,
                        self.DATASET_INFO.output_classes,
                    )),
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
                    feature_names=self.DATASET_INFO.feature_names,
                    output_class_names=list(map(
                        lambda x: x.name,
                        self.DATASET_INFO.output_classes,
                    )),
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
                    feature_names=self.DATASET_INFO.feature_names,
                    output_class_names=list(map(
                        lambda x: x.name,
                        self.DATASET_INFO.output_classes,
                    )),
                ),
            )

        raise ValueError(
            f'Given rule extractor "{extractor_name}" is not a valid rule '
            f'extracting algorithm. Valid modes are "REM-D" or '
            f'"pedagogical".'
        )

    def _initialize_directories(self):
        """
        Helper method to initialize all the directories we will need for our
        experiment run.
        """
        # Create main output directory first
        if os.path.exists(self.experiment_dir):
            if self._forced_rerun:
                logging.warning(
                    "Running in overwrite mode. This means that some previous "
                    f"results in output directory {self.experiment_dir} may be "
                    "overwritten by this run."
                )
            else:
                logging.warning(
                    f"Given experiment directory {self.experiment_dir} exists. "
                    "This means that we may use any results from previous "
                    "experiment runs to avoid retraining/computing in this "
                    "run. If this directory contains any results from "
                    "different experiments that are not compatible with this "
                    "one, or if you want to rerun the entire experiment from "
                    "scratch, please call this script with the --forced_rerun "
                    "argument."
                )
        os.makedirs(self.experiment_dir, exist_ok=True)
        # Create directory: cross_validation/<n>_folds/
        os.makedirs(self.N_FOLD_CV_DP, exist_ok=True)
        # Create directory: <n>_folds/rule_extraction/<rulemode>/
        os.makedirs(self.N_FOLD_RULE_EX_MODE_DP, exist_ok=True)
        # Create directory: <n>_folds/rule_extraction/<rulemode>/rules_extracted
        os.makedirs(self.N_FOLD_RULES_DP, exist_ok=True)
        # Create directory: <n>_folds/trained_models
        os.makedirs(self.N_FOLD_MODELS_DP, exist_ok=True)

        # Initialise split indices file for folds
        os.makedirs(
            pathlib.Path(self.N_FOLD_CV_SPLIT_INDICES_FP).parent,
            exist_ok=True,
        )

        # Initialise split indices for train/test split
        os.makedirs(
            pathlib.Path(self.NN_INIT_SPLIT_INDICES_FP).parent,
            exist_ok=True,
        )

    def _initialize_data(self):
        """
        Helper method for us to initialize our data set loading for the
        given experiment.
        """

        # Read our dataset
        self.X, self.y, _ = self.DATASET_INFO.read_data(self.DATA_FP)

        # Split our data into two groups of train and test data in general
        self.data_split, _ = self.serializable_stage(
            target_file=self.NN_INIT_SPLIT_INDICES_FP,
            execute_fn=lambda: stratified_k_fold_split(
                X=self.X,
                y=self.y,
                random_state=self.RANDOM_SEED,
                test_size=self.PERCENT_TEST_DATA,
                n_folds=1,
            ),
            serializing_fn=split_serializer,
            deserializing_fn=split_deserializer,
            stage_name="data_split",
        )

        # And do the same but now for the folds that we will use for training
        self.fold_split, _ = self.serializable_stage(
            target_file=self.N_FOLD_CV_SPLIT_INDICES_FP,
            execute_fn=lambda: stratified_k_fold_split(
                X=self.X,
                y=self.y,
                random_state=self.RANDOM_SEED,
                test_size=self.PERCENT_TEST_DATA,
                n_folds=self.N_FOLDS,
            ),
            serializing_fn=split_serializer,
            deserializing_fn=split_deserializer,
            stage_name="fold_split",
        )

    def get_fold_data(self, fold):
        """
        Gets the train and test data for the requested fold.

        :param uint fold: The fold id (zero indexed) whose dataset we want to
            obtain.

        :returns Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
            (X_train, y_train, X_test, y_test) with the data corresponding to
            this split.

        :raises ValueError: when given fold is not a valid fold for the dataset
            handled by this manager.
        """
        if (fold > len(self.fold_split)) or (fold <= 0):
            raise ValueError(
                f'We obtained a request for split of fold {fold} however we '
                f'have only split the dataset into {len(self.fold_split)} '
                f'folds with first fold being fold 1.'
            )
        train_indices, test_indices = self.fold_split[fold - 1]
        X_train, y_train = self.X[train_indices], self.y[train_indices]
        X_test, y_test = self.X[test_indices], self.y[test_indices]

        # And do any required preprocessing we may need to do to the
        # data now that it has been partitioned (to avoid information
        # leakage in data-dependent preprocessing passes)
        if self.DATASET_INFO.preprocessing:
            X_train, y_train, X_test, y_test = self.DATASET_INFO.preprocessing(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        return X_train, y_train, X_test, y_test

    def get_train_split(self):
        """
        Gets the train and test data for entire dataset treated as a whole.

        :returns Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
            (X_train, y_train, X_test, y_test) with the data corresponding to
            our experiment's dataset.
        """
        train_indices, test_indices = self.data_split[0]
        X_train, y_train = self.X[train_indices], self.y[train_indices]
        X_test, y_test = self.X[test_indices], self.y[test_indices]

        # And do any required preprocessing we may need to do to the
        # data now that it has been partitioned (to avoid information
        # leakage in data-dependent preprocessing passes)
        if self.DATASET_INFO.preprocessing:
            X_train, y_train, X_test, y_test = self.DATASET_INFO.preprocessing(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        return X_train, y_train, X_test, y_test

    def serializable_stage(
        self,
        target_file,
        execute_fn,
        # Empty serialization function by default
        serializing_fn=lambda result, path: result,
        # Trivial deserializing function by default
        deserializing_fn=lambda path: None,
        stage_name=None,
    ):
        """
        Method for blocking the execution of a call if that call has been done
        before and its result has been serialized. Used for loading checkpoints
        when not running in "forced_rerun" mode and obtaining their results
        if available.

        This is intended to be called for stages that are SEQUENTIAL in nature.
        In that spirit, if one method fails to hit and use the given checkpoint
        file, then all subsequent calls to this method will fail to use the
        cache as well.

        If an error occurs while deserializing a given file, then we will
        move to recomputing it using our execution function.

        :param str target_file: The serialization target file we will try to
            deserialize and load our data from using `deserializing_fn` or
            we will serialize the result of `execute_fn` into if it does not
            exit (or we are running using a forced rerun).
        :param Fun() -> Any execute_fn: The function to be executed when we
            did not positively found the checkpoint file. This function should
            take no arguments.
        :param Fun(Any, str) -> Any serializing_fn: Function taking the result
            of running execute_fn() and a file name and serializes that result
            into the given file. Returns the value we should return after
            serialization as the result of this function.
        :param Fun(str) -> Any deserializing_fn:  The deserializing function
            to be used if given target file was found so that we can load up
            our result.

        :returns Tuple[Any, bool]: A tuple (result, hit) where is the output of
            serializing_fn(execute_fn(), _) if we did not find `target_file` and
            otherwise it is deserializing_fn(target_file). `hit` is True if we
            managed to deserialize the target file and False otherwise.
        """

        if os.path.exists(target_file) and (not self._forced_rerun) and (
            (stage_name is None) or
            (self._start_rerun_stage is None) or
            (stage_name != self._start_rerun_stage)
        ):
            # Then time to deserialize it and return it to the user
            try:
                logging.debug(f'We hit the cache for "{target_file}"')
                return deserializing_fn(target_file), True
            except Exception as e:
                # Then at this point we will simply recompute it as the
                # deserialization did not work
                logging.debug(f'We error {e} during deserialization...')

        # Else we will run the whole thing from scratch and we WILL FORCE ALL
        # FUTURE CALLS TO ALSO DO THE SAME THING
        self._forced_rerun = True
        result = execute_fn()

        # Serialize it and allow for further data processing (if any)
        return serializing_fn(result, target_file), False

