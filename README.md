# REM: Rule Extraction Methods for Neural Networks
Code base for the experimentation and methods for REM: Rule Extraction Methods, A Breast Cancer Case Study. This repository will include reproduction steps for the experiments presented in the papers as well as packaged functions to use the described rule extraction algorithm in any custom Sequential Keras model.

## Credits

TODO: link to paper and authors

A lot of the code and structure of the project is strongly based on the work in this repository: https://github.com/sumaiyah/DNN-RE.


## Setup
For you to be able to run recreate the experiments and use the rule extraction algorithm, you will need the following requirements:
- `python` 3.5 – 3.8
- `pip` 19.0 or later
- `R` 4.* needs to be installed and accessible in your machine. We use `rpy2` to wrap and run `R`'s C5.0 algorithm.

Once you have installed R, you will also need to have the following packages installed in R:
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
                         [--force_rerun] [-d]

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
  --force_rerun, -f     If set and we are given as output directory the
                        directory of a previous run, then we will overwrite
                        any previous work and redo all computations.
                        Otherwise, we will attempt to use as much as we can
                        from the previous run.
  -d, --debug           starts debug mode in our program.
```

One can run the tool by manually inputing the dataset information as command-line arguments or by providing a YAML config as the following example:
```yaml
# The directory of our training data. Can be a path relative to the caller or
# an absolute path.
dataset_file: "../DNN-RE-data-new/MB-1004-GE-2Hist/data.csv"

# The name of our dataset. Must be one of the data sets supported by our
# experiment runners.
dataset_name: 'MB-1004-GE-2Hist'

# Number of split folds for our training. If not provided, then it will default
# to a single fold.
n_folds: 5

# Our neural network training hyper-parameters
hyperparameters:
    # The batch size we will use for training.
    batch_size: 16
    # Number of epochs to run our model for
    epochs: 50
    # Now many hidden layers we will use and how many activations in each
    layer_units: [128, 64]
    # The activation use in between hidden layers. Can be any valid Keras
    # activation function. If not provided, it defaults to "tanh"
    activation: "tanh"
    # The last activation used in our model. Used to define the type of
    # categorical loss we will use. If not provided, it defaults to "sigmoid".
    last_activation: "sigmoid"
    # The type of loss we will use. Must be one of
    # ["sofxmax_xentr", "sigmoid_xentr"]. If not provided, we will use the
    # given last layer's activation to obtain the corresponding loss.
    loss_function: "softmax_xentr"

# How many trials we will attempt for finding our best initialisation. If not
# provided, then we will use a random initialisation at train time.
initialisation_trials: 5

# The rule extractor we will use. If not provided, it defaults to REM-D.
rule_extractor: "REM-D"

# Where are we dumping our results. If not provided, it will default to the same
# directory as the one containing the dataset.
output_dir: "experiment_results"
```

In this example, we are indicating the path where we are we storing our `MB-1004-GE-2Hist` dataset and what hyper-parameters we want to use for our neural network.

You can then use this to run the experiment as follows:
```bash
python run_experiment.py --config experiment_config.yaml
```

If run successfully, then you should see an output similar to this one:
```bash
$ python run_experiment.py --config experiment_config.yaml
Test accuracy for fold 1/5 is 0.829, AUC is 0.828, and majority class accuracy is 0.918
Test accuracy for fold 2/5 is 0.814, AUC is 0.805, and majority class accuracy is 0.91
Test accuracy for fold 3/5 is 0.9, AUC is 0.734, and majority class accuracy is 0.911
Test accuracy for fold 4/5 is 0.923, AUC is 0.868, and majority class accuracy is 0.907
Test accuracy for fold 5/5 is 0.938, AUC is 0.685, and majority class accuracy is 0.915
Training fold model 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:34<00:00,  6.97s/it]
Done extracting rules from neural network: 100%|█████████████████████████████████████████████████████████████████████████████████████████▉| 5.9999999999999964/6 [06:23<00:00, 64.00s/it]
Done extracting rules from neural network: 100%|█████████████████████████████████████████████████████████████████████████████████████████▉| 5.999999999999999/6 [10:00<00:00, 100.10s/it]
Done extracting rules from neural network: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.0/6 [04:17<00:00, 42.91s/it]
Done extracting rules from neural network: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 6.0000000000000036/6 [04:12<00:00, 42.13s/it]
[WARNING] Found an empty set of rules for class 0 and layer 0
[WARNING] Found an empty set of rules for class 0 and layer 0
Done extracting rules from neural network: 100%|██████████████████████████████████████████████████████████████████████████████████████████▉| 5.999999999999998/6 [03:09<00:00, 31.55s/it]
+------+-------------+--------+----------------+-----------------------+------------------------+
| Fold | NN Accuracy | NN AUC | REM-D Accuracy | Extraction Time (sec) | Extraction Memory (MB) |
+------+-------------+--------+----------------+-----------------------+------------------------+
|  0   |    0.8289   | 0.8283 |     0.8643     |        384.2171       |       6688.6728        |
|  1   |    0.8142   | 0.8046 |     0.9086     |        600.7569       |        892.9505        |
|  2   |    0.8997   | 0.7343 |     0.9145     |        257.5909       |        568.0148        |
|  3   |    0.9233   | 0.8676 |     0.8761     |        252.9575       |        532.0765        |
|  4   |    0.9379   | 0.6848 |     0.9142     |        189.4459       |        459.6419        |
| avg  |    0.8808   | 0.7839 |     0.8955     |        336.9937       |       1828.2713        |
+------+-------------+--------+----------------+-----------------------+------------------------+
~~~~~~~~~~~~~~~~~~~~ Experiment successfully terminated after 1727.306 seconds ~~~~~~~~~~~~~~~~~~~~
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
