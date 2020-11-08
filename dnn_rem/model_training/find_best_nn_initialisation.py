"""
Find best neural network initialisation

1. Load train test split
2. Build 5 Neural Networks with different initialisations using best
   hyper-parameters.
3. Perform rule extraction on these 5 networks
4. network with smallest ruleset, save that initialisation
"""
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from dnn_rem.evaluate_rules.evaluate import evaluate
from . import split_data
from .build_and_train_model import build_and_train_model
from dnn_rem.experiment_runners import dnn_re


def run(X, y, manager):
    train_index, test_index = split_data.load_split_indices(
        file_path=manager.NN_INIT_SPLIT_INDICES_FP,
    )

    # Split data
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # Save information about nn initialisation
    if not os.path.exists(manager.NN_INIT_RE_RESULTS_FP):
        pd.DataFrame(data=[], columns=['run']).to_csv(
            manager.NN_INIT_RE_RESULTS_FP,
            index=False,
        )

    # Path to trained neural network
    model_file_path = os.path.join(manager.TEMP_DIR, 'model.h5')

    # Smallest ruleset i.e. total number of rules
    smallest_ruleset_size = np.float('inf')
    smallest_ruleset_acc = 0
    best_init_index = 0

    with tqdm(
        range(manager.INITIALISATION_TRIALS),
        desc="Finding best initialization"
    ) as pbar:
        for i in pbar:
            pbar.set_description(
                f'Testing initialisation {i + 1}/'
                f'{manager.INITIALISATION_TRIALS}'
            )

            # Build and train nn put it in temp/
            nn_accuracy, nn_auc = build_and_train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                manager=manager,
                model_file_path=model_file_path,
                with_best_initilisation=False,
            )
            if logging.getLogger().getEffectiveLevel() not in [
                logging.ERROR,
                logging.WARNING,
            ]:
                pbar.write(
                    f"Test accuracy for initialisation {i + 1}/"
                    f"{manager.INITIALISATION_TRIALS} is {nn_accuracy} and "
                    f"AUC is {nn_auc}"
                )

            # Extract rules
            _, rules, re_time, _ = dnn_re.run(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                manager=manager,
                model_file_path=model_file_path,
                # Lower our verbosity level for the purpose of not having the
                # bar being printed twice
                verbosity=(
                    logging.WARNING if (
                        logging.getLogger().getEffectiveLevel() != logging.DEBUG
                    ) else logging.DEBUG
                ),
            )

            # Save labels to labels.csv:
            # label - True data labels
            label_data = {
                'id': test_index,
                'true_labels': y_test,
            }
            # label - Neural network data labels. Use NN to predict X_test
            nn_model = tf.keras.models.load_model(model_file_path)
            nn_predictions = np.argmax(nn_model.predict(X_test), axis=1)
            label_data['nn_labels'] = nn_predictions
            # label - Rule extraction labels
            rule_predictions = rules.predict(X_test)
            label_data[
                f'rule_{manager.RULE_EXTRACTOR.mode}_labels'
            ] = rule_predictions
            pd.DataFrame(data=label_data).to_csv(manager.LABEL_FP, index=False)

            # Save rule extraction time and memory usage
            results_df = pd.read_csv(manager.NN_INIT_RE_RESULTS_FP)
            results_df.loc[i, 'run'] = i
            results_df.loc[i, 're_time (sec)'] = re_time

            re_results = evaluate(rules=rules, manager=manager)
            results_df.loc[i, 'nn_acc'] = nn_accuracy
            results_df.loc[i, 're_acc'] = re_results['acc']
            results_df.loc[i, 're_fid'] = re_results['fid']
            results_df.loc[i, 'rules_num'] = sum(
                re_results['n_rules_per_class']
            )

            results_df["run"] = results_df["run"].astype(int)
            results_df["nn_acc"] = results_df["nn_acc"].round(3)
            results_df["re_acc"] = results_df["re_acc"].round(3)
            results_df["re_fid"] = results_df["re_fid"].round(3)
            results_df["re_time (sec)"] = results_df["re_time (sec)"].round(3)
            results_df["rules_num"] = results_df["rules_num"]

            results_df = results_df[[
                "run",
                "nn_acc",
                "re_acc",
                "re_fid",
                "re_time (sec)",
                "rules_num"
            ]]

            results_df.to_csv(manager.NN_INIT_RE_RESULTS_FP, index=False)

            # If this initialisation extracts a smaller ruleset - save it
            ruleset_size = sum(re_results['n_rules_per_class'])
            if (ruleset_size < smallest_ruleset_size) or (
                (ruleset_size == smallest_ruleset_size) and
                (re_results['acc'] > smallest_ruleset_acc)
            ):
                smallest_ruleset_size = ruleset_size
                smallest_ruleset_acc = re_results['acc']
                best_init_index = i

                # Save initilisation as best_initialisation.h5
                tf.keras.models.load_model(
                    os.path.join(manager.TEMP_DIR, 'initialisation.h5')
                ).save(manager.BEST_NN_INIT_FP)

        pbar.set_description("Done finding best initialisation")
    logging.debug(
        f'Found neural network with the best initialisation '
        f'at index {best_init_index} and saved in path '
        f'{manager.BEST_NN_INIT_FP}'
    )
