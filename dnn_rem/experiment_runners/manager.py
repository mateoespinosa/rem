"""
Class for managing directory hierarchies when running experiments with the
rule generation algorithm.
"""

from collections import namedtuple
import os
import pathlib
import shutil
import tempfile

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

        # Validate our provided config
        self.validate_config_object(config)

        # Now time to build all the directory variables we will need to
        # construct our experiment
        self.DATASET_INFO = dataset_configs.get_data_configuration(
            dataset_name=config["dataset_name"],
        )

        self.RULE_EXTRACTOR = self.get_rule_extractor(
            config.get("rule_extractor", "rem_d")
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
        self.NN_INIT_RE_RESULTS_FP = os.path.join(
            nn_init_dir,
            're_results.csv'
        )
        self.BEST_NN_INIT_FP = os.path.join(
            nn_init_dir,
            'best_initialisation.h5'
        )

        # Store temporary files during program execution. If an explicit
        # temporary directory is not provided, then we will make our own
        self.TEMP_DIR = config.get(
            'tmp_dir',
            tempfile.mkdtemp()
        )
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        self.LABEL_FP = os.path.join(self.TEMP_DIR, 'labels.csv')

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
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Safe exit code from our experiment execution. Make sure
        we remove our temporary folders as soon as we are out.
        """
        print("~~~~~~~ EXPERIMENT CONCLUDED ~~~~~~~")
        self.stream.close()
        if os.path.exists(self.TEMP_DIR) and os.path.isdir(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)

    @staticmethod
    def validate_config_object(config):
        """
        Validates the given 'config' deserialized as a given YAML object.
        Makes sure all required fields are provided and that they have
        sensible values. If this is not the case, then a ValueError will be
        thrown.

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


    @staticmethod
    def get_rule_extractor(extractor_name):
        name = extractor_name.lower()
        if name == "rem-d":
            return RuleExMode(
                mode='REM-D',
                run=rem_d,
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
