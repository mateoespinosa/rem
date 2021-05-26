# ECLAIRE: Efficient CLAuse-wIse Rule Extraction
Main repository for work our work in ECLAIRE, a novel decompositional rule extraction method for DNNs. This repository will include reproduction steps for all experiments presented in  as well as packaged functions to use the described rule extraction algorithms in any custom Keras model.

## Credits

A lot of the code and structure of the project is based on the work by Shams et al. made available in: [https://github.com/ZohrehShams/IntegrativeRuleExtractionMethodology](https://github.com/ZohrehShams/IntegrativeRuleExtractionMethodology). This has been summarized in the publication ["REM: An Integrative Rule Extraction Methodology for Explainable Data Analysis in Healthcare
"](https://www.biorxiv.org/content/10.1101/2021.01.22.427799v2.abstract) by Shams et al.


## Setup
For you to be able to run recreate the experiments and use the rule extraction algorithm, you will need the following requirements:
- `python` 3.5 – 3.8
- `pip` 19.0 or later
- `R` 4.* needs to be installed and accessible in your machine. We use `rpy2` to wrap and run `R`'s C5.0 algorithm.

Once you have installed R, you will also need to have the following packages installed in R:
- `C50`
- `Cubist`
- `reshape2`
- `plyr`
- `Rcpp`
- `stringr`
- `stringi`
- `magrittr`
- `partykit`
- `Formula`
- `libcoin`
- `mvtnorm`
- `inum`

If you have all of these, then you can install our code as a Python package using pip as follows:
```python
python setup.py install --user
```

This will install all required the dependencies for you as well as the entire project. Please note that this may take some time if you are missing some of the heavy dependencies we require (e.g TensorFlow).

**Important Note**: depending on your `python` distribution and environment (specially if you are using `pyenv` or a virtual environment), you may have to add `--prefix=` (nothing after the equality) to get this installation to work for you.

If you want to test that it works, try running the provided script `run_experiment.py` as follows:
```bash
python run_experiment.py --help
```
and you should see a clear help message without any errors and/or warnings.

## Running Experiments
For our experiments we will use the data located in https://github.com/sumaiyah/DNN-RE-data. You can run recreate several of our cross-validation experiments using the provided executable `run_experiment.py` which offers the following options:
```bash
$python run_experiment.py --help
usage: run_experiment.py [-h] [--config file.yaml] [--n_folds N]
                         [--dataset_name name] [--dataset_file data.cvs]
                         [--rule_extractor name] [--grid_search]
                         [--output_dir path] [--randomize]
                         [--force_rerun {all,data_split,fold_split,grid_search,nn_train,rule_extraction}]
                         [--profile] [-d]
                         [-p param_name=value param_name=value]

Runs cross validation experiment with the given rule extraction method and our neural network training.

optional arguments:
  -h, --help            show this help message and exit
  --config file.yaml, -c file.yaml
                        initial configuration YAML file for our experiment's
                        setup.
  --n_folds N, -n N     how many folds to use for our data partitioning.
  --dataset_name name   name of the dataset to be used for training.
  --dataset_file data.cvs
                        comma-separated-valued file containing our training
                        data.
  --rule_extractor name
                        name of the extraction algorithm to be used to
                        generate our rule set.
  --grid_search         whether we want to do a grid search over our model's
                        hyperparameters. If the results of a previous grid
                        search are found in the provided output directory,
                        then we will use those rather than starting a grid
                        search from scratch. This means that turning this flag
                        on will overwrite any hyperparameters you provide as
                        part of your configuration (if any).
  --output_dir path, -o path
                        directory where we will dump our experiment's results.
                        If not given, then we will use the same directory as
                        our dataset.
  --randomize, -r       If set, then the random seeds used in our execution
                        will not be fixed and the experiment will be
                        randomized. By default, otherwise, all experiments are
                        run with the same seed for reproducibility.
  --force_rerun {all,data_split,fold_split,grid_search,nn_train,rule_extraction}, -f {all,data_split,fold_split,grid_search,nn_train,rule_extraction}
                        If set and we are given as output directory the
                        directory of a previous run, then we will overwrite
                        any previous work starting from the provided stage
                        (and all subsequent stages of the experiment) and redo
                        all computations. Otherwise, we will attempt to use as
                        much as we can from the previous run.
  --profile             prints out profiling statistics of the rule-extraction
                        in terms of low-level calls used for the extraction
                        method.
  -d, --debug           starts debug mode in our program.
  -p param_name=value param_name=value, --param param_name=value param_name=value
                        Allows the passing of a config param that will
                        overwrite anything passed as part of the config file
                        itself.
```

One can run the tool by manually inputing the dataset information as command-line arguments or by providing a YAML config containing the experiment parameterization as the following example:
```yaml
# The directory of our training data. Can be a path relative to the caller or
# an absolute path.
dataset_file: "../DNN-RE-data-new/XOR/data.csv"

# The name of our dataset. Must be one of the data sets supported by our
# experiment runners.
dataset_name: 'XOR'

# Whether or not we want our experiment runner to overwrite results from
# previous runs found in the given output directory or we want to reuse them
# instead.
# If not provided, or set to null, then we will NOT overwrite any
# previous results and use them as checkpoints to avoid double work in this
# experiment.
# Otherwise, it must be one of
#    - "all"
#    - "data_split"
#    - "fold_split"
#    - "grid_search"
#    - "nn_train"
#    - "rule_extraction"
# to indicate the stage in which we will start rewriting previous results. If
# such a specific stage is provided, then all subsequent stages will be also
# overwritten (following the same order as the list above)
force_rerun: null

# Number of split folds for our training. If not provided, then it will default
# to a single fold.
n_folds: 5

# Our neural network training hyper-parameters
hyperparameters:
    # The batch size we will use for training.
    batch_size: 16
    # Number of epochs to run our model for
    epochs: 150
    # Now many hidden layers we will use and how many activations in each
    layer_units: [64, 32, 16]
    # The activation use in between hidden layers. Can be any valid Keras
    # activation function. If not provided, it defaults to "tanh"
    activation: "tanh"
    # The last activation used in our model. Used to define the type of
    # categorical loss we will use. If not provided, it defaults to the
    # corresponding last activation for the given loss function.
    last_activation: "softmax"
    # The type of loss we will use. Must be one of
    # ["softmax_xentr", "sigmoid_xentr"]. If not provided, we will use the
    # given last layer's activation to obtain the corresponding loss if it was
    # provided. Otherwise, we will default to softmax xentropy.
    loss_function: "softmax_xentr"
    # The learning rate we will use for our Adam optimizer. If not provided,
    # then it will be defaulted to 0.001
    learning_rate: 0.001
    # Dropout rate to use in layer in between last hidden layer and last layer
    # If 0, then no dropout is done. This is the probability that a given
    # activation will be dropped.
    dropout_rate: 0
    # Frequency of skip connections in the form of additions. If zero, then
    # no skip connections are made
    skip_freq: 0


    ###########################################
    # Early Stopping Parameters
    ###########################################

    # Early stopping parameters. Only valid if patience is greater than 0.
    early_stopping_params:
        validation_percent: 0.1
        patience: 0
        monitor: "loss"
        min_delta: 0.001
        mode: 'min'

    # Optimizer parameters
    optimizer_params:
        decay_rate: 1
        decay_steps: 1

    ###########################################
    # Compression Parameters
    ###########################################

    # The algorithm used for compressing this model. If not compression required
    # then set this to null.
    # Needs to be one of: ["weight-magnitude", "unit-magnitude", null]
    compress_mechanism: null

    # Parameters specific to the selected compression algorithm. Only relevant
    # if compress_mechanism is not null.
    compression_params:
        # How many prune->retraining epochs we will run for the given algorithm
        pruning_epochs: 20

        # What is the desired target sparsity? Sparsity can be weight sparsity
        # or activation sparsity depending on the used compression mechanism
        target_sparsity: 0.75

        # We will perform compression using annealing which will enable us to
        # have a starting sparsity target and build up towards the required
        # end sparsity
        initial_sparsity: 0.0

# How many subprocesses to use to evaluate our ruleset in the testing data
evaluate_num_workers: 6

# The rule extractor we will use. If not provided, it defaults to ECLAIRE.
# Must be one of [
#   "REM-D",
#   "ECLAIRE", (or equivalently "eREM-D")
#   "cREM-D",
#   "Pedagogical",
#   "Clause-REM-D",
#   "DeepRED",
#   "REM-T",
#   "sREM-D"
# ]
rule_extractor: "ECLAIRE"

# And any parameters we want to provide to the extractor for further
# tuning/experimentation. This is dependent on the used extractor
extractor_params:
    # An integer indicating how many decimals should we truncate our thresholds
    # to. If null, then no truncation will happen.
    # For original REM-D: set to null
    threshold_decimals: null

    # The winnow parameter to use for C5 for intermediate rulesets.
    # Must be a boolean.
    # For original REM-D: set to True
    winnow_intermediate: True

    # The winnow parameter to use for C5 for the last ruleset generation (which
    # depends on the actual features)
    # Must be a boolean.
    # For original REM-D: set to True
    winnow_features: True

    # The minimum number of cases for a split in C5. Must be a positive integer
    min_cases: 2

    # What is the maximum number of training samples to use when building trees
    # If a float, then this is seen as a fraction of the dataset
    # If 0 or null, then no restriction is applied.
    # For original REM-D: set to null
    max_number_of_samples: null

    ###########################################
    # REM-T Parameters
    ###########################################

    # The name of the algorithm used to extract the decision trees if using
    # REM-T
    # Must be one of: [C5.0, random_forest, cart]
    tree_extraction_algorithm_name: "C5.0"

    # Whether or not we perform posthoc CCP pruning when growing decision trees
    ccp_prune: true

    # Number of estimators to use in random forest if using REM-T
    estimators: 10

    # The maximum depth allowed for a single tree when using REM-T with
    # CART and/or random_forest
    tree_max_depth: 10

    ###########################################
    # ECLAIRE Parameters
    ###########################################

    # The rule extraction algorithm used to change intermediate rule sets in
    # ECLAIRE to be a function of input feature activations only.
    # Must be one of ["C5.0", "CART", "random_forest"]
    # If not given then C5.0 is used
    final_algorithm_name: "C5.0"

    # The rule extraction algorithm used to generate intermediate rule sets in
    # ECLAIRE.
    # Must be one of ["C5.0", "CART", "random_forest"]
    # If not given then C5.0 is used
    intermediate_algorithm_name: "C5.0"

    # The number of processes (i.e. workers) to use when extracting rules from
    # our ruleset. Make sure to pick a reasonable number depending on the number
    # of cores you have
    # For original REM-D: set to 1
    num_workers: 6

    # Number of blocks to partition our network in when performing intermediate
    # rule extraction
    # For original REM-D: set to 1
    block_size: 1

    # Bagging trials for reducing the variance in our ruleset generation
    # Must be an integer greater than or equal to 1
    # For original REM-D: set to 1
    trials: 1

    # Whether or not we preemptively remove redundant clauses
    # For original REM-D: set to False
    preemptive_redundant_removal: False

    # Fraction of lowest-activating activations we will drop when building up
    # our predictors using an intermediate layer's activations
    # For original REM-D: set to 1
    top_k_activations: 1

    # What percent of intermediate rules we will drop using our the given
    # ranking mechanism
    # For original REM-D: set to 0
    intermediate_drop_percent: 0

    # We can anneal the dropping rate so that later layers drop less terms than
    # those in the start. We do this using a linear schedule starting with
    # initial_drop_percent and ending with intermediate_drop_percent
    # If null and intermediate_drop_percent is greater than zero, then we will
    # set this to intermediate_drop_percent as well.
    # For original REM-D: set to null
    initial_drop_percent: null

    # Rule scoring mechanism to be used for dropping intermediate ruleset
    # activations. Only relevant if intermediate_drop_percent is greater than
    # zero.
    # Has to be one of ["Majority", "Accuracy", "HillClimb", "Confidence"]
    rule_score_mechanism: "Confidence"

    # Whether or not when we iterate over terms, we merge their negations into
    # the set of terms to avoid extracting them twice
    # For original REM-D: set to False
    merge_repeated_terms: True

    # Minimum confidence required for a given rule to be included in the
    # a produced ruleset
    # For original REM-D: set to 0
    min_confidence: 0

    # Maximum number of rules allowed in any intermediate ruleset generated
    # by the decompositional algorithm. If ruleset has more rules than this
    # number, then we will drop the lowest ranking rules until we have a rule
    # set as with exactly max_intermediate_rules
    # If null, then no rule pruning happens
    max_intermediate_rules: null

    # Whether or not intermediate rule elimination happens at a per-class basis
    # or not.
    per_class_elimination: true

    # The initial value of min cases to for later layers in the network.
    # If 0 or null, then we will use the same as min_cases throughout.
    # Otherwise we will do a linear interpolation where the end number of min
    # cases is given by min_cases
    # For original REM-D: set to null
    initial_min_cases: null

    # The number of min cases used at the end of processing all intermediate
    # layers (if annealing is done).
    # If null or 0 then we will use the same value as min_cases
    intermediate_end_min_cases: null

    # Whether or not we will use class weights in C5.0 when changing variables
    # in intermediate rule sets
    balance_classes: false

    # Maximum depth of tree in rule extractor used to generate intermediate rule
    # sets if CART and/or random_forest is used.
    intermediate_tree_max_depth: 15

    # Maximum depth of tree used to change variables in intermediate rule sets
    # if CART and/or random_forest is used.
    final_tree_max_depth: 10

    # If True, then ECLAIRE will include input feature activations as part of
    # the set of activations it can use to extract intermediate rule sets from.
    ecclectic: False



# We can specify which mechanism we will obtain a score a given rule given our
# training data and its class. When classifying a new point, we will assign
# scores to every rule and then pick the class corresponding to the ruleset with
# the highest average score for all triggered rules of that set. We currently
# support the following scoring functions:
#   - "Majority": Rule majority voting will be done to determine the output
#                 class. This means every rule has a score of 1.
#   - "Accuracy": accuracy of each rule in the training set will be used to
#                 score each rule.
#   - "HillClimb": HillClimb scoring function based on the training set will be
#                  used to score each rule.
#   - "Confidence": the confidence level of each rule when generating the
#                   ruleset will be used as its score function.
# If not given, we will default to "Majority".
rule_score_mechanism: "Majority"

# If we wan to drop the lowest `percent` rules after scoring, then you can do
# so by modifying this argument here which will drop them after scoring and
# before evaluation. This must be a real number in [0, 1]
rule_elimination_percent: 0

# Where are we dumping our results. If not provided, it will default to the same
# directory as the one containing the dataset.
output_dir: "experiment_results"

# Parameters to be used during our grid search
grid_search_params:
    # Whether or not to perform grid-search
    enable: False
    # The metric we will optimize over during our grid-search. Must be one of
    # ["accuracy", "auc"]
    metric_name: "accuracy"
    # Batch sizes to be used during training
    batch_sizes: [16, 32]
    # Training epochs to use for our DNNs
    epochs: [50, 100, 150]
    # Learning rates to try in our optimizer
    learning_rates: [0.001, 0.0001]
    # The sizes to try for each hidden layer
    layer_sizes: [[128, 64, 32], [64, 32]]
    # Activations to use between hidden layers. Must be valid Keras activations
    activations: ["tanh", "elu"]
    # The amount of dropout to use between hidden layers and the last layer
    dropout_rates: [0, 0.2]
    # Finally, the loss function to use for training
    loss_functions: ["softmax_xentr", "sigmoid_xentr"]

```

In this example, we are indicating the path where we are we storing our `MB-1004-GE-ER` dataset and what hyper-parameters we want to use for our neural network.

You can then use this to run the experiment as follows:
```bash
python run_experiment.py --config experiment_config.yaml
```

If run successfully, then you should see an output similar to this one:
```bash
$ python run_experiment.py --config experiment_config.yaml
Test accuracy for fold 1/5 is 0.98, AUC is 0.979, and majority class accuracy is 0.607                                                                                                                                                        
Test accuracy for fold 2/5 is 0.935, AUC is 0.935, and majority class accuracy is 0.598                                                                                                                                                       
Test accuracy for fold 3/5 is 0.985, AUC is 0.986, and majority class accuracy is 0.576                                                                                                                                                       
Test accuracy for fold 4/5 is 0.955, AUC is 0.955, and majority class accuracy is 0.58                                                                                                                                                        
Test accuracy for fold 5/5 is 0.975, AUC is 0.973, and majority class accuracy is 0.567                                                                                                                                                       
Training fold model 5/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:35<00:00,  7.01s/it]
Done extracting intermediate rulesets: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.11it/s]
Substituting rules for layer 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.50it/s]
[INFO] Performing prediction over 200 test samples using 79 rules and 6 effective processes.
[INFO] Rule set test accuracy for fold 1/5 is 0.875, AUC is 0.87, fidelity is 0.875, and size of rule set is 79
Done extracting intermediate rulesets: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.83it/s]
Substituting rules for layer 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.42it/s]
[INFO] Performing prediction over 200 test samples using 76 rules and 6 effective processes.
[INFO] Rule set test accuracy for fold 2/5 is 0.93, AUC is 0.928, fidelity is 0.925, and size of rule set is 76
Done extracting intermediate rulesets: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.80it/s]
Substituting rules for layer 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.30it/s]
[INFO] Performing prediction over 200 test samples using 77 rules and 6 effective processes.
[INFO] Rule set test accuracy for fold 3/5 is 0.955, AUC is 0.953, fidelity is 0.94, and size of rule set is 77
Done extracting intermediate rulesets: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.59it/s]
Substituting rules for layer 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.10it/s]
[INFO] Performing prediction over 200 test samples using 119 rules and 6 effective processes.
[INFO] Rule set test accuracy for fold 4/5 is 0.915, AUC is 0.911, fidelity is 0.9, and size of rule set is 119
Done extracting intermediate rulesets: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.87it/s]
Substituting rules for layer 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.15it/s]
[INFO] Performing prediction over 200 test samples using 84 rules and 6 effective processes.
[INFO] Rule set test accuracy for fold 5/5 is 0.915, AUC is 0.91, fidelity is 0.93, and size of rule set is 84
+------+---------------+---------------+------------------+---------------+------------------+-----------------------+------------------------+---------------+---------------------+----------------+
| Fold |  NN Accuracy  |     NN AUC    | ECLAIRE Accuracy |  ECLAIRE AUC  | ECLAIRE Fidelity | Extraction Time (sec) | Extraction Memory (MB) |  Ruleset Size | Average Rule Length |   # of Terms   |
+------+---------------+---------------+------------------+---------------+------------------+-----------------------+------------------------+---------------+---------------------+----------------+
|  1   |      0.98     |     0.979     |      0.875       |      0.87     |      0.875       |         3.189         |        598.283         |       79      |        3.063        |      174       |
|  2   |     0.935     |     0.935     |       0.93       |     0.928     |      0.925       |         3.396         |        427.146         |       76      |         3.0         |      146       |
|  3   |     0.985     |     0.986     |      0.955       |     0.953     |       0.94       |         3.619         |        255.616         |       77      |        2.766        |      152       |
|  4   |     0.955     |     0.955     |      0.915       |     0.911     |       0.9        |         4.122         |        326.681         |      119      |        3.445        |      268       |
|  5   |     0.975     |     0.973     |      0.915       |      0.91     |       0.93       |         3.899         |        719.342         |       84      |        2.893        |      178       |
| avg  | 0.966 ± 0.019 | 0.966 ± 0.018 |  0.918 ± 0.026   | 0.914 ± 0.027 |  0.914 ± 0.024   |     3.645 ± 0.335     |   465.414 ± 171.383    | 87.0 ± 16.236 |    3.033 ± 0.229    | 183.6 ± 43.953 |
+------+---------------+---------------+------------------+---------------+------------------+-----------------------+------------------------+---------------+---------------------+----------------+
~~~~~~~~~~~~~~~~~~~~ Experiment successfully terminated after 58.47 seconds ~~~~~~~~~~~~~~~~~~~~
```

The default cross-validation experiment will do the following:
  1. Split our dataset into training/test datasets and further split our training data into the number of requested folds.
  2. Train one neural network for each fold of data.
  3. Extract rules for each neural network we trained.
  4. Compare the performance of the neural network and the extracted ruleset in our test data.
  5. Dump all the results, statistics, and details of the experiment in the provided directory following the file hierarchy described below.

If the output directory exists and contains data from a previous run, our runner will attempt to reuse it as much as possible to avoid recomputing things. If you do not want that behavior, please make sure to use different directories for different runs or call the script with the `-f` flag in it.

Please note that if a configuration file is provided and command-line arguments are also provided, then the ones given in the command-line will always take precedence over their counterparts in the config file. The intent of this behavior is to speed up different iterations in experiments.

### Experiment Folder structure
Once an experiment is instantiated, we will generate a file structure containing all the results of the experiment as follows:
```markdown
`<output-dir>`
    `cross_validation/` - contains data from cross-validated rule_extraction

    ​   `<n>_folds/` - contains data from e.g. 10_folds/

    ​       `rule_extraction/`

    ​           `<rule_ex_mode>/` - e.g. eclaire, pedagogical, rem-d

    ​               `results.csv` - results for rule extraction using that mode

    ​               `rules_extracted/` - saving the rules extracted to disk

    ​                   `fold_<n>.rules` - Pickle object of our rule set
                       `file_<n>.rules.txt` - Human-readable version of the rules.

    ​       `trained_models/` - trained neural network models for each fold

    ​           `fold_<n>_model.h5`

    ​       `data_split_indices.txt` - indices for n stratified folds of the data

    `grid_search_results.txt` - results from neural network hyper-parameter grid search

    `data_split_indices.txt` - indices of data split for train and test data

    `config.yaml` - copy of the configuration file used for this experiment for recreation purposes
```
At the end of the experiment, you should expect to find this file structure in the provided results output directory.

If no output directory is provided, then we will use the same directory as the one containing our dataset.


## Using Custom Models

You can use the ECLAIRE algorithm as described in the paper with any custom Keras model (REM-D can be used as well but only on sequential models). To do this, you can import the following method once you have installed this package as instructed in the setup:

```python
from dnn_rem import eclaire
# Read data from some source
X_train, y_train = ...
# Train a Keras model on this data
keras_model = ...
# Extract rules from that trained model
ruleset = eclaire.extract_rules(keras_model, X_train)
# And try and make predictions using this ruleset
X_test = ...
y_pred = ruleset.predict(X_test)

# You can see the learned ruleset by printing it
print(ruleset)
```
