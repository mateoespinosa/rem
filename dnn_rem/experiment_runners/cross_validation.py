from prettytable import PrettyTable
import logging
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from dnn_rem.evaluate_rules.evaluate import evaluate
from dnn_rem.model_training.build_and_train_model import load_model


def _deserialize_rules(path):
    # Helper method to deserialize our rules and the resource
    # consumption it took.
    with open(path, 'rb') as f:
        return pickle.load(f)


def _serialize_rules(result, path):
    # Helper method to seserialize our rules and the resource
    # consumption it took into a given file
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return result


def cross_validate_re(manager):
    # We will generate a pretty table for the end result so that it can
    # be pretty-printed at the end of the experiment and visually reported to
    # the user
    table = PrettyTable()
    table.field_names = [
        "Fold",
        "NN Accuracy",
        'NN AUC',
        f"{manager.RULE_EXTRACTOR.mode} Accuracy",
        f"{manager.RULE_EXTRACTOR.mode} AUC",
        f"{manager.RULE_EXTRACTOR.mode} Fidelity",
        "Extraction Time (sec)",
        "Extraction Memory (MB)",
        "Rulset Size",
        "Average Rule Length",
    ]
    averages = np.array([0.0] * (len(table.field_names) - 1))
    results_df = pd.DataFrame(data=[], columns=['fold'])

    # Extract rules from model from each fold
    for fold in range(manager.N_FOLDS):

        ########################################################################
        ## Neural Network Evaluation
        ########################################################################

        # Get train and test data folds
        X_train, y_train, X_test, y_test = manager.get_fold_data(fold)

        # Path to neural network model for this fold
        model_file_path = manager.n_fold_model_fp(fold)
        nn_model = load_model(model_file_path)
        _, nn_auc, nn_accuracy, majority_class = nn_model.evaluate(
            X_test,
            tf.keras.utils.to_categorical(y_test),
            verbose=(
                1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                else 0
            ),
        )

        ########################################################################
        ## Rule Extraction Evaluation
        ########################################################################

        # Path to extracted rules from that fold
        extracted_rules_file_path = manager.n_fold_rules_fp(fold)

        # Run our rule extraction only if it has not been done in the past
        # through a sequential checkpoint
        (rules, re_time, re_memory), _ = manager.serializable_stage(
            target_file=extracted_rules_file_path,
            execute_fn=lambda: manager.resource_compute(
                function=manager.RULE_EXTRACTOR.run,
                model=nn_model,
                train_data=X_train,
            ),
            serializing_fn=_serialize_rules,
            deserializing_fn=_deserialize_rules,
            stage_name="rule_extraction",
        )

        # Serialize a human readable version of the rules always for inspection
        with open(extracted_rules_file_path + ".txt", 'w') as f:
            for rule in rules:
                f.write(str(rule) + "\n")

        logging.debug(
            f'Evaluating rules extracted from '
            f'fold {fold + 1}/{manager.N_FOLDS}...'
        )

        re_results = evaluate(
            rules=rules,
            X_test=X_test,
            y_test=y_test,
            high_fidelity_predictions=np.argmax(
                nn_model.predict(X_test),
                axis=1
            ),
            num_labels=manager.DATASET_INFO.n_outputs,
        )

        ########################################################################
        ## Table writing and saving
        ########################################################################

        # Same some of this information into our dataframe
        results_df.loc[fold, 'fold'] = fold
        results_df.loc[fold, 'nn_accuracy'] = nn_accuracy
        results_df.loc[fold, 'nn_auc'] = nn_auc
        results_df.loc[fold, 'majority_class'] = majority_class
        results_df.loc[fold, 're_time (sec)'] = re_time
        results_df.loc[fold, 're_memory (MB)'] = re_memory
        results_df.loc[fold, 're_acc'] = re_results['acc']
        results_df.loc[fold, 're_fid'] = re_results['fid']
        results_df.loc[fold, 're_auc'] = re_results['auc']
        results_df.loc[fold, 'output_classes'] = str(
            re_results['output_classes']
        )
        results_df.loc[fold, 're_n_rules_per_class'] = str(
            re_results['n_rules_per_class']
        )
        results_df.loc[fold, 'n_overlapping_features'] = str(
            re_results['n_overlapping_features']
        )
        results_df.loc[fold, 'min_n_terms'] = str(
            re_results['min_n_terms']
        )
        results_df.loc[fold, 'max_n_terms'] = str(
            re_results['max_n_terms']
        )
        results_df.loc[fold, 'av_n_terms_per_rule'] = str(
            re_results['av_n_terms_per_rule']
        )

        logging.debug(
            f"Rule extraction for fold {fold} took a total of "
            f"{re_time} sec "
            f"and {re_memory} MB to obtain "
            f"testing accuracy {re_results['acc']} compared to the accuracy "
            f"of the neural network {nn_accuracy}."
        )

        # Fill up our pretty table
        avg_rule_length = np.array(re_results['av_n_terms_per_rule'])
        avg_rule_length *= np.array(re_results['n_rules_per_class'])
        avg_rule_length = sum(avg_rule_length)
        avg_rule_length /= sum(re_results['n_rules_per_class'])
        num_rules = sum(re_results['n_rules_per_class'])
        new_row = [
            round(nn_accuracy, manager.ROUNDING_DECIMALS),
            round(nn_auc, manager.ROUNDING_DECIMALS),
            round(re_results['acc'], manager.ROUNDING_DECIMALS),
            round(re_results['auc'], manager.ROUNDING_DECIMALS),
            round(re_results['fid'], manager.ROUNDING_DECIMALS),
            round(re_time,  manager.ROUNDING_DECIMALS),
            round(re_memory, manager.ROUNDING_DECIMALS),
            num_rules,
            round(avg_rule_length, manager.ROUNDING_DECIMALS),
        ]
        table.add_row([fold + 1] + new_row)

        # And accumulate this last row unto our average
        averages += np.array(new_row) / manager.N_FOLDS

        # Finally, log this int the progress bar if not on quiet mode to get
        # some feedback
        logging.info(
            f'Rule set test accuracy for fold {fold + 1}/{manager.N_FOLDS} '
            f'is {round(re_results["acc"], 3)}, AUC is '
            f'{round(re_results["auc"], 3)}, and size of rule set is '
            f'{num_rules}'
        )

    # Now that we are done, let's serialize our dataframe for further analysis
    results_df.to_csv(manager.N_FOLD_RESULTS_FP, index=False)

    # Finally, let's include an average column:
    table.add_row(
        ["avg"] +
        list(map(lambda x: round(x,  manager.ROUNDING_DECIMALS), averages))
    )

    # And display our results as a pretty table for the user to inspect quickly
    if logging.getLogger().getEffectiveLevel() not in [
        logging.WARNING,
        logging.ERROR,
    ]:
        print(table)
    # And always serialize our table results into a pretty-printed txt file
    with open(manager.SUMMARY_FILE, 'w') as f:
        f.write(table.get_string())
