# REM: Rule Extraction Methods for Neural Networks
Code base for the experimentation and methods for [REM: Rule Extraction Methods, A Breast Cancer Case Study](TODO: ADD LINK HERE). This repository will include reproduction steps for the experiments presented in the papers as well as packaged functions to use the described rule extraction algorithm in any custom Sequential Keras model.

## Setup
For you to be able to run recreate the experiments and use the rule extraction algorithm, you will need the following requirements:
- Python 3.5 – 3.8
- pip 19.0 or later
- R 4.* needs to be installed and accessible

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
- `mutnorm`
- `inum`

If you have all of these, then you can install our code as a Python package using pip as follows:
```python
python setup.py install --user
```
This will install all required the dependencies for you as well as the entire project. Please note that this may take some time if you are missing some of the heavy dependencies we require (e.g TensorFlow).

If you want to test that it works, try running the provided script `run_experiment.py` as follows:
```bash
python run_experiment.py --help
```
and you should see a clear help message without any errors and/or warnings.

## Running Experiments
For our experiments we will use the data located in https://github.com/sumaiyah/DNN-RE-data. You can run recreate several experiments using the provided executable `run_experiment.py` which offers the following options:
```bash
$python run_experiment.py --help 
usage: run_experiment.py [-h] [--config file.yaml] [--folds N]
                         [--initialisation_trials N] [--dataset_name name]
                         [--dataset_file data.cvs] [--rule_extractor name]
                         [--grid_search] [--output_dir path] [--tmp_dir path]
                         [-d]

Process some integers.

optional arguments:
  -h, --help            show this help message and exit
  --config file.yaml, -c file.yaml
                        initial configuration YAML file for our experiment's
                        setup.
  --folds N, -f N       how many folds to use for our data partitioning.
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
  --tmp_dir path        temporary directory to use for scratch work. If not
                        provided, then we will instantiate our own temporary
                        directory to be removed at the end of the program's
                        execution.
  -d, --debug           starts debug mode in our program.
```

One can run the tool by manually inputing the dataset information as command-line arguments or by providing a YAML config as the following example:
```yaml
# The directory of our training data
dataset_file: "../DNN-RE-data-new/MB-1004-GE-2Hist/data.csv"
# The name of our dataset
dataset_name: 'MB-1004-GE-2Hist'
# Number of split folds for our training
n_folds: 5
# Our neural network training hyper-parameters
hyperparameters:
    batch_size: 16
    epochs: 50
    layer_units: [128, 64]
# How many trials we will attempt for finding our best initialisation
initialisation_trials: 5
# The rule extractor we will use (if not given, we assume DeepRED_C5 is used)
rule_extractor: "DeepRED_C5"
# Where are we dumping our results. If not given, then it will be the same
# directory as the one containing the dataset..
output_dir: "experiment_results"
```

In this example, we are indicating the path where we are we storing our `MB-1004-GE-2Hist` dataset and what hyper-parameters we want to use for our neural network.

You can then use this to run the experiment as follows:
```python
python run_experiment.py --config experiment_config.yaml
```

Please note that if a configuration file is provided and command-line arguments are also provided, then the ones given in the command-line will always take precedence over their counterparts in the config file. The intent of this behavior is to speed up different iterations in experiments.

### Experiment Folder structure
Once an experiment is instantiated, we will generate a file structure containing all the results of the experiment as follows:
```markdown
`<output-dir>`
    `cross_validation/` - contains data from cross-validated rule_extraction

    ​   `<n>_folds/` - contains data from e.g. 10_folds/

    ​       `rule_extraction/`

    ​           `<rule_ex_mode>/` - e.g. pedagogical, decomp

    ​               `results.csv` - results for rule extraction using that mode

    ​               `rules_extracted/` - saving the rules extracted to disk

    ​                   `fold_<n>.rules` - Pickle object of our rule set
                       `file_<n>.rules.txt` - Human-readable version of the rules.

    ​       `trained_models/` - trained neural network models for each fold

    ​           `fold_<n>_model.h5`

    ​       `data_split_indices.txt` - indices for n stratified folds of the data

    `neural_network_initialisation/` - contains data from finding the best neural network initialisation

    ​   `re_results.csv` - rule extraction results from each of the initialisations 

    ​   `grid_search_results.txt` - results from neural network hyperparameter grid search

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
```
