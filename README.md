# REM: Rule Extraction Methods for Neural Networks
Code base for the experimentation and methods for REM: Rule Extraction Methods, A Breast Cancer Case Study. This repository will include reproduction steps for the experiments presented in the papers as well as packaged functions to use the described rule extraction algorithm in any custom Sequential Keras model.

## Credits

A lot of the code and structure of the project is strongly based on the work in this repository: https://github.com/sumaiyah/DNN-RE. This has been summarized in the publication ["REM: An Integrative Rule Extraction Methodology for Explainable Data Analysis in Healthcare
"](https://www.medrxiv.org/content/10.1101/2021.01.25.21250459v1.full) by Shams et al.


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
                         [--initialisation_trials N] [--dataset_name name]
                         [--dataset_file data.cvs] [--rule_extractor name]
                         [--grid_search] [--output_dir path] [--randomize]
                         [--force_rerun {all,data_split,fold_split,grid_search,initialisation_trials,nn_train,rule_extraction}]
                         [-d]

Process some integers.

optional arguments:
  -h, --help            show this help message and exit
  --config file.yaml, -c file.yaml
                        initial configuration YAML file for our experiment's
                        setup.
  --n_folds N, -n N     how many folds to use for our data partitioning.
  --initialisation_trials N, -i N
                        how many initialisations for our model's initial
                        parameters do we want to try before running our
                        training. If less than or equal to one, then we will
                        simply use a random initialisation.
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
  --force_rerun {all,data_split,fold_split,grid_search,initialisation_trials,nn_train,rule_extraction}, -f {all,data_split,fold_split,grid_search,initialisation_trials,nn_train,rule_extraction}
                        If set and we are given as output directory the
                        directory of a previous run, then we will overwrite
                        any previous work starting from the provided stage
                        (and all subsequent stages of the experiment) and redo
                        all computations. Otherwise, we will attempt to use as
                        much as we can from the previous run.
  -d, --debug           starts debug mode in our program.
```

One can run the tool by manually inputing the dataset information as command-line arguments or by providing a YAML config containing the experiment parameterization as the following example:
```yaml
# The directory of our training data. Can be a path relative to the caller or
# an absolute path.
dataset_file: "../DNN-RE-data-new/MB-GE-ER/data.csv"

# The name of our dataset. Must be one of the data sets supported by our
# experiment runners.
dataset_name: 'MB-GE-ER'

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
#    - "initialisation_trials"
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
    epochs: 100
    # Now many hidden layers we will use and how many activations in each
    layer_units: [128, 128]
    # The activation use in between hidden layers. Can be any valid Keras
    # activation function. If not provided, it defaults to "tanh"
    activation: "elu"
    # The last activation used in our model. Used to define the type of
    # categorical loss we will use. If not provided, it defaults to the
    # corresponding last activation for the given loss function.
    last_activation: "softmax"
    # The type of loss we will use. Must be one of
    # ["sofxmax_xentr", "sigmoid_xentr"]. If not provided, we will use the
    # given last layer's activation to obtain the corresponding loss if it was
    # provided. Otherwise, we will default to softmax xentropy.
    loss_function: "softmax_xentr"
    # The learning rate we will use for our Adam optimizer. If not provided,
    # then it will be defaulted to 0.001
    learning_rate: 0.0001
    # Dropout rate to use in layer in between last hidden layer and last layer
    # If 0, then no dropout is done. This is the probability that a given
    # activation will be dropped.
    dropout_rate: 0

# How many trials we will attempt for finding our best initialisation. If not
# provided or less than or equal to 1, then we will use a random initialisation
# at train time.
initialisation_trials: 1

# If we are looking for a best initialisation, then we also need to provide
# a metric to optimize over. This can be one of [accuracy", "auc"]
initialisation_trial_metric: "accuracy"

# The rule extractor we will use. If not provided, it defaults to REM-D.
rule_extractor: "REM-D"

# And any parameters we want to provide to the extractor for further
# tuning/experimentation. This is dependent on the used extractor
extractor_params:
    # An integer indicating how many decimals should we truncate our thresholds
    # to. If null, then no truncation will happen.
    threshold_decimals: 6
    # The winnow parameter to use for C5. Must be a boolean.
    winnow: True
    # The minimum number of cases for a split in C5. Must be a positive integer
    min_cases: 15
    # The number of processes (i.e. workers) to use when extracting rules from
    # our ruleset. Make sure to pick a reasonable number depending on the number
    # of cores you have
    num_workers: 4

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
Test accuracy for fold 1/5 is 0.944, AUC is 0.949, and majority class accuracy is 0.762
Test accuracy for fold 2/5 is 0.98, AUC is 0.976, and majority class accuracy is 0.756
Test accuracy for fold 3/5 is 0.952, AUC is 0.958, and majority class accuracy is 0.76
Test accuracy for fold 4/5 is 0.96, AUC is 0.948, and majority class accuracy is 0.764
Test accuracy for fold 5/5 is 0.96, AUC is 0.937, and majority class accuracy is 0.764
Training fold model 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.22it/s]
[INFO] Rule set test accuracy for fold 1/5 is 0.924, AUC is 0.888, and size of rule set is 150
[INFO] Rule set test accuracy for fold 2/5 is 0.97, AUC is 0.958, and size of rule set is 88
[INFO] Rule set test accuracy for fold 3/5 is 0.904, AUC is 0.872, and size of rule set is 98
[INFO] Rule set test accuracy for fold 4/5 is 0.914, AUC is 0.875, and size of rule set is 28
[INFO] Rule set test accuracy for fold 5/5 is 0.909, AUC is 0.897, and size of rule set is 196
+------+-------------+--------+----------------+-----------+----------------+-----------------------+------------------------+-------------+---------------------+
| Fold | NN Accuracy | NN AUC | REM-D Accuracy | REM-D AUC | REM-D Fidelity | Extraction Time (sec) | Extraction Memory (MB) | Rulset Size | Average Rule Length |
+------+-------------+--------+----------------+-----------+----------------+-----------------------+------------------------+-------------+---------------------+
|  1   |    0.944    | 0.949  |     0.924      |   0.888   |     0.919      |         87.806        |        1149.617        |     150     |        5.307        |
|  2   |     0.98    | 0.976  |      0.97      |   0.958   |      0.97      |        111.085        |        456.966         |      88     |        5.011        |
|  3   |    0.952    | 0.958  |     0.904      |   0.872   |     0.922      |        106.309        |         426.48         |      98     |        4.031        |
|  4   |     0.96    | 0.948  |     0.914      |   0.875   |     0.914      |         62.085        |         292.33         |      28     |        3.643        |
|  5   |     0.96    | 0.937  |     0.909      |   0.897   |     0.904      |        106.863        |        857.591         |     196     |        6.429        |
| avg  |    0.959    | 0.954  |     0.924      |   0.898   |     0.926      |         94.83         |        636.597         |    112.0    |        4.884        |
+------+-------------+--------+----------------+-----------+----------------+-----------------------+------------------------+-------------+---------------------+
```

The default cross-validation experiment will do the following:
  1. Split our dataset into training/test datasets and further split our training data into the number of requested folds.
  2. Find the best initialisation for the weights of our neural network architecture.
  3. Train one neural network for each fold of data.
  4. Extract rules for each neural network we trained.
  5. Compare the performance of the neural network and the extracted ruleset in our test data.
  6. Dump all the results, statistics, and details of the experiment in the provided directory following the file hierarchy described below.

If the output directory exists and contains data from a previous run, our runner will attempt to reuse it as much as possible to avoid recomputing things. If you do not want that behavior, please make sure to use different directories for different runs or call the script with the `-f` flag in it.

Please note that if a configuration file is provided and command-line arguments are also provided, then the ones given in the command-line will always take precedence over their counterparts in the config file. The intent of this behavior is to speed up different iterations in experiments.

### Experiment Folder structure
Once an experiment is instantiated, we will generate a file structure containing all the results of the experiment as follows:
```markdown
`<output-dir>`
    `cross_validation/` - contains data from cross-validated rule_extraction

    ​   `<n>_folds/` - contains data from e.g. 10_folds/

    ​       `rule_extraction/`

    ​           `<rule_ex_mode>/` - e.g. pedagogical, rem-d

    ​               `results.csv` - results for rule extraction using that mode

    ​               `rules_extracted/` - saving the rules extracted to disk

    ​                   `fold_<n>.rules` - Pickle object of our rule set
                       `file_<n>.rules.txt` - Human-readable version of the rules.

    ​       `trained_models/` - trained neural network models for each fold

    ​           `fold_<n>_model.h5`

    ​       `data_split_indices.txt` - indices for n stratified folds of the data

    `neural_network_initialisation/` - contains data from finding the best neural network initialisation

    ​   `re_results.csv` - rule extraction results from each of the initialisations

    ​   `grid_search_results.txt` - results from neural network hyper-parameter grid search

    ​   `data_split_indices.txt` - indices of data split for train and test data

    ​   `best_initialisation.h5` - best neural network initialisation i.e. the one that generated the smallest ruleset
```
At the end of the experiment, you should expect to find this file structure in the provided results output directory.

If no output directory is provided, then we will use the same directory as the one containing our dataset.


## Using Custom Models

You can use the REM-D algorithm as described in the paper with any custom Keras sequential model (note that sequentiality is a requirement for now). To do this, you can import the following method once you have installed this package as instructed in the setup:

```python
from dnn_rem import rem_d
# Read data from some source
X_train, y_train = ...
# Train a Keras Sequential model on this data
keras_model = ...
# Extract rules from that trained model
ruleset = rem_d.extract_rules(keras_model, X_train)
# And try and make predictions using this ruleset
X_test = ...
y_pred = ruleset.predict(X_test)

# You can see the learned ruleset by printing it
print(rulset)
```
