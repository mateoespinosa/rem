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
from dnn_rem.utils.resources import resource_compute
from .build_and_train_model import run_train_loop, load_model


def run(manager):
    # Split data. Note that we WILL NOT use our test data at all for this given
    # that that would imply data leakage into our training loop. Ideally this
    # should be done with a validation set rather than with the training set
    # itself.
    logging.info("Finding best initialisation")
    X_train, y_train, _, _ = manager.get_train_split()

    # The metric name to optimize over for our given initializations
    metric_name = manager.BEST_INIT_METRIC_NAME
    if metric_name == "accuracy":
        # Then we rewrite it as "acc" as that's what we use in for stats
        # collection in here
        metric_name = "acc"

    # Save information about nn initialisation
    if not os.path.exists(manager.NN_INIT_RE_RESULTS_FP):
        pd.DataFrame(data=[], columns=['run']).to_csv(
            manager.NN_INIT_RE_RESULTS_FP,
            index=False,
        )

    # Smallest ruleset i.e. total number of rules
    smallest_ruleset_size = np.float('inf')
    smallest_ruleset_metric = 0
    best_init_index = 0
    best_model = None

    # We will store all results in a table for later analysis
    results_df = pd.DataFrame(data=[], columns=['fold'])
    with tqdm(
        range(manager.INITIALISATION_TRIALS),
        desc="Finding best initialization"
    ) as pbar:
        for i in pbar:
            pbar.set_description(
                f'Testing initialisation {i + 1}/'
                f'{manager.INITIALISATION_TRIALS}'
            )

            ####################################################################
            ## NEURAL NETWORK TRAIN + EVALUATION
            ####################################################################

            # Build and train nn put it in temp/
            model, nn_loss, nn_accuracy, nn_auc, maj_class_acc = run_train_loop(
                X_train=X_train,
                y_train=y_train,
                # Validate on our training data itself. In the near future,
                # use a validation set for this instead
                X_test=X_train,
                y_test=y_train,
                manager=manager,
            )

            ####################################################################
            ## RULE EXTRACTION + EVALUATION
            ####################################################################

            rules, re_time, _ = resource_compute(
                function=manager.RULE_EXTRACTOR.run,
                model=model,
                train_data=X_train,
                # Lower our verbosity level for the purpose of not having the
                # bar being printed twice
                verbosity=(
                    logging.WARNING if (
                        logging.getLogger().getEffectiveLevel() != logging.DEBUG
                    ) else logging.DEBUG
                ),
            )

            re_results = evaluate(
                rules=rules,
                # Validate on our training data itself
                X_test=X_train,
                y_test=y_train,
                high_fidelity_predictions=np.argmax(
                    model.predict(X_train),
                    axis=1
                ),
            )

            ####################################################################
            ## BEST MODEL SELECTION
            ####################################################################

            # If this initialisation extracts a smaller ruleset - save it
            ruleset_size = sum(re_results['n_rules_per_class'])

            if logging.getLogger().getEffectiveLevel() not in [
                logging.ERROR,
                logging.WARNING,
            ]:
                pbar.write(
                    f"Test accuracy for initialisation {i + 1}/"
                    f"{manager.INITIALISATION_TRIALS} is "
                    f"{round(nn_accuracy, 3)}, "
                    f"AUC is {round(nn_auc, 3)}, and majority class accuracy "
                    f"is {round(maj_class_acc, 3)}. Number of rules extracted "
                    f"was {ruleset_size}."
                )

            if (ruleset_size < smallest_ruleset_size) or (
                (ruleset_size == smallest_ruleset_size) and
                (re_results[metric_name] > smallest_ruleset_metric)
            ):
                smallest_ruleset_size = ruleset_size
                smallest_ruleset_metric = re_results[metric_name]
                best_init_index = i

                # Keep our best model
                best_model = model

            ####################################################################
            ## RESULT SUMMARY
            ####################################################################

            # Save rule extraction time and memory usage
            results_df.loc[i, 'run'] = i
            results_df.loc[i, 're_time (sec)'] = re_time
            results_df.loc[i, 'nn_acc'] = nn_accuracy
            results_df.loc[i, 're_acc'] = re_results['acc']
            results_df.loc[i, 're_auc'] = re_results['auc']
            results_df.loc[i, 're_fid'] = re_results['fid']
            results_df.loc[i, 'rules_num'] = sum(
                re_results['n_rules_per_class']
            )
            results_df["run"] = results_df["run"].astype(int)
            results_df["nn_acc"] = results_df["nn_acc"].round(3)
            results_df["re_acc"] = results_df["re_acc"].round(3)
            results_df["re_auc"] = results_df["re_auc"].round(3)
            results_df["re_fid"] = results_df["re_fid"].round(3)
            results_df["re_time (sec)"] = results_df["re_time (sec)"].round(3)
            results_df["rules_num"] = results_df["rules_num"]

            results_df = results_df[[
                "run",
                "nn_acc",
                "re_acc",
                "re_auc",
                "re_fid",
                "re_time (sec)",
                "rules_num"
            ]]

        # Serialize table of results
        results_df.to_csv(manager.NN_INIT_RE_RESULTS_FP, index=False)
        pbar.set_description("Done finding best initialisation")

    logging.debug(
        f'Found neural network with the best initialisation '
        f'at index {best_init_index}.'
    )
    return best_model
