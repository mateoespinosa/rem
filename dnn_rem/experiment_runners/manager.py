"""
Class for managing directory hierarchies when running experiments with the
rule generation algorithm.
"""

from collections import namedtuple
import os
import pathlib
import shutil
import tempfile
import time
import tracemalloc

from . import dataset_configs
from dnn_rem.extract_rules.rem_d import extract_rules as rem_d
from dnn_rem.extract_rules.pedagogical import extract_rules as pedagogical


# Algorithm used for Rule Extraction
RuleExMode = namedtuple('RuleExMode', ['mode', 'run'])


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

    def __init__(self, config):

        # Some hidden state for management purposes only
        self._start_time = time.time()

        # Validate our provided config
        self.validate_config_object(config)

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
        self.ROUNDING_DECIMALS = config.get("rounding_decimals", 4)

        # And build our rule extractor
        self.RULE_EXTRACTOR = self.get_rule_extractor(
            config.get("rule_extractor", "rem_d")
        )

        # Where all our results will be dumped. If not provided as part of the
        # experiment's config, then we will use the same parent directory as the
        # datafile we are using
        experiment_dir = config.get(
            "output_dir",
            pathlib.Path(self.DATA_FP).parent
        )

        # <dataset_name>/cross_validation/<n>_folds/
        cross_val_dir = os.path.join(experiment_dir, 'cross_validation')
        self.N_FOLD_CV_DP = f'{cross_val_dir}{self.N_FOLDS}_folds/'
        self.N_FOLD_CV_SPLIT_INDICIES_FP = os.path.join(
            self.N_FOLD_CV_DP,
            'data_split_indices.txt'
        )
        self.N_FOLD_CV_SPLIT_X_train_data_FP = (
            lambda fold: os.path.join(
                self.N_FOLD_CV_DP,
                f'fold_{fold}_X_train.npy',
            )
        )
        self.N_FOLD_CV_SPLIT_y_train_data_FP = (
            lambda fold: os.path.join(
                self.N_FOLD_CV_DP,
                f'fold_{fold}_y_train.npy',
            )
        )
        self.N_FOLD_CV_SPLIT_X_test_data_FP = (
            lambda fold: os.path.join(
                self.N_FOLD_CV_DP,
                f'fold_{fold}_X_test.npy',
            )
        )
        self.N_FOLD_CV_SPLIT_y_test_data_FP = (
            lambda fold: os.path.join(
                self.N_FOLD_CV_DP,
                f'fold_{fold}_y_test.npy',
            )
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
            experiment_dir,
            'neural_network_initialisation'
        )
        self.NN_INIT_GRID_RESULTS_FP = os.path.join(
            nn_init_dir,
            'grid_search_results.txt'
        )
        self.NN_INIT_SPLIT_INDICES_FP = os.path.join(
            nn_init_dir,
            'data_split_indices.txt'
        )
        self.NN_INIT_RE_RESULTS_FP = os.path.join(
            nn_init_dir,
            're_results.csv'
        )
        self.BEST_NN_INIT_FP = os.path.join(
            nn_init_dir,
            'best_initialisation.h5'
        )

        n_fold_rules_dir = os.path.join(
            self.N_FOLD_CV_DP,
            'rule_extraction/MOD_DecisionTree',
        )
        self.N_FOLD_RESULTS_DT_FP = os.path.join(
            n_fold_rules_dir,
            'results.csv',
        )
        self.N_FOLD_RULES_DT_DP = os.path.join(
            n_fold_rules_dir,
            'rules_extracted',
        )
        self.n_fold_rules_DT_fp = (
            lambda fold: os.path.join(
                self.N_FOLD_RULES_DT_DP,
                f'fold_{fold}.rules',
            )
        )

        n_fold_rules_dir = os.path.join(
            self.N_FOLD_CV_DP,
            'rule_extraction/MOD_RandomForest',
        )
        self.N_FOLD_RESULTS_RF_FP = os.path.join(
            n_fold_rules_dir,
            'results.csv',
        )
        self.N_FOLD_RULES_RF_DP = os.path.join(
            n_fold_rules_dir,
            'rules_extracted',
        )
        n_fold_rules_RF_fp = (
            lambda fold: os.path.join(
                self.N_FOLD_RULES_RF_DP,
                f'fold_{fold}.rules',
            )
        )

        n_fold_rules_dir = os.path.join(
            self.N_FOLD_CV_DP,
            'rule_extraction/MOD_DT_Combined',
        )
        self.N_FOLD_RESULTS_DT_COMB_FP = os.path.join(
            n_fold_rules_dir,
            'results.csv'
        )
        self.N_FOLD_RULES_DT_COMB_DP = os.path.join(
            n_fold_rules_dir,
            'rules_extracted'
        )
        n_fold_rules_DT_COMB_fp = (
            lambda fold: os.path.join(
                self.N_FOLD_RULES_DT_COMB_DP,
                f'fold_{fold}.rules',
            )
        )

        n_fold_rules_dir = os.path.join(
            self.N_FOLD_CV_DP,
            'rule_extraction/MOD_RF_Combined',
        )
        self.N_FOLD_RESULTS_RF_COMB_FP = os.path.join(
            n_fold_rules_dir,
            'results.csv',
        )
        self.N_FOLD_RULES_RF_COMB_DP = os.path.join(
            n_fold_rules_dir,
            'rules_extracted/',
        )
        self.n_fold_rules_RF_COMB_fp = (
            lambda fold: os.path.join(
                self.N_FOLD_RULES_RF_COMB_DP,
                f'fold_{fold}.rules',
            )
        )

        self.N_FOLD_RULES_REMAINING_DP = \
            lambda reduction_percentage: os.path.join(
                self.N_FOLD_RULE_EX_MODE_DP,
                f'rules_remaining_after_{reduction_percentage}_reduction',
            )
        self.n_fold_rules_fp_remaining = (
            lambda path, fold: (
                lambda reduction_percentage: os.path.join(
                    path(reduction_percentage),
                    f'fold_{fold}_remaining.rules'
                )
            )
        )
        self.N_FOLD_RESULTS_FP_REMAINING = \
            lambda reduction_percentage: os.path.join(
                self.N_FOLD_RULE_EX_MODE_DP,
                f'results_{reduction_percentage}_reduction.csv'
            )
        self.rules_fp_remaining = lambda reduction_percentage: os.path.join(
            N_FOLD_RULES_REMAINING_DP(reduction_percentage),
            "fold.rules",
        )

    def __enter__(self):
        """
        Enter code. Nothing to do here specifically as all the setup is
        done in our initializer
        """
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Safe exit code from our experiment execution. Make sure
        we remove our temporary folders as soon as we are out.
        """
        print(
            "~" * 20,
            "Experiment successfully terminated after",
            round(time.time() - self._start_time, 3),
            "seconds",
            "~" * 20,
        )

    def resource_compute(self, function, *args, **kwargs):
        """
        Evaluates function(*args, **kwargs) and returns the time and memory
        consumption of that evaluation.

        :param fun function: An arbitrary function to run with arguments
            *args and **kwargs.
        :param List[Any] args: Positional arguments for the provided function.
        :param Dictionary[str, Any] kwargs: Key-worded arguments to the provided
            function.

        :returns Tuple[Any, float, float]: Tuple (call_results, time, memory)
            where `results` are the results of calling
            function(*args, **kwargs), time is the time it took for executing
            that function in seconds, and memory is the total memory consumption
            for that function in MB.

        """

        # Start our clock and memory handlers
        start_time = time.time()
        tracemalloc.start()

        # Run the function
        result = function(*args, **kwargs)

        # And compute memory usage
        memory, peak = tracemalloc.get_traced_memory()
        # Tracemalloc reports memory in Kibibytes, so let's change it to MB
        memory = memory * (1024 / 1000000)
        tracemalloc.stop()

        return result, (time.time() - start_time), memory

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

    def get_rule_extractor(self, extractor_name):
        name = extractor_name.lower()
        if name == "rem-d":
            loss_function = self.HYPERPARAMS.get(
                "loss_function",
                "sigmoid_xentr",
            )
            last_activation = self.HYPERPARAMS.get("last_activation", "sigmoid")
            # We set the last activation to None here if it is going to be
            # be included in the network itself. Otherwise, we request our
            # rule extractor to explicitly perform the activation on the last
            # layer as this was merged into the loss function
            last_activation = (
                last_activation if last_activation in loss_function else None
            )
            return RuleExMode(
                mode='REM-D',
                run=lambda *args, **kwargs: rem_d(
                    *args,
                    **kwargs,
                    last_activation=last_activation,
                )
            )

        if name == "pedagogical":
            return RuleExMode(
                mode='pedagogical',
                run=pedagogical,
            )

        raise ValueError(
            f'Given rule extractor "{extractor_name}" is not a valid rule '
            f'extracting algorithm. Valid modes are "REM-D" or '
            f'"pedagogical".'
        )
