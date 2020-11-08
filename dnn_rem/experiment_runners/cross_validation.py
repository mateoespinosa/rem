import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf


from dnn_rem.evaluate_rules.evaluate import evaluate
from dnn_rem.model_training.split_data import load_split_indices
from . import dnn_re


def cross_validate_re(
    X,
    y,
    manager,
):
    # Extract rules from model from each fold
    for fold in range(manager.N_FOLDS):
        # Path to extracted rules from that fold
        extracted_rules_file_path = manager.n_fold_rules_fp(fold)

        # Get train and test data folds
        train_index, test_index = load_split_indices(
            manager.N_FOLD_CV_SPLIT_INDICIES_FP,
            fold_index=fold,
        )

        # Path to neural network model for this fold
        model_file_path = manager.n_fold_model_fp(fold)
        _, _, nn_accuracy = tf.keras.models.load_model(
            model_file_path
        ).evaluate(
            X[test_index],
            tf.keras.utils.to_categorical(y[test_index]),
            verbose=(
                1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG \
                else 0
            )
        )

        # Extract rules
        _, rules, re_time, re_memory = dnn_re.run(
            X_train=X[train_index],
            y_train=y[train_index],
            X_test=X[test_index],
            y_test=y[test_index],
            manager=manager,
            model_file_path=model_file_path,
        )

        # Save rules extracted
        logging.debug(
            f'Saving fold {fold + 1}/{manager.N_FOLDS} rules extracted...'
        )
        # Serialize both as a pickle object as well as human readable file.
        # In the near future, we will be able to serialize this to an actual
        # lightweight language to express these expressions
        with open(extracted_rules_file_path, 'wb') as f:
            pickle.dump(rules, f)
        with open(extracted_rules_file_path + ".txt", 'w') as f:
            for rule in rules:
                f.write(str(rule) + "\n")

        logging.debug('done')

        # Save rule extraction time and memory usage
        logging.debug(
            f'Saving fold {fold + 1}/{manager.N_FOLDS} results...'
        )
        # Initialize empty results file
        if not fold:
            pd.DataFrame(data=[], columns=['fold']).to_csv(
                manager.N_FOLD_RESULTS_FP,
                index=False,
            )

        results_df = pd.read_csv(manager.N_FOLD_RESULTS_FP)
        results_df.loc[fold, 'fold'] = fold
        results_df.loc[fold, 'nn_accuracy'] = nn_accuracy
        results_df.loc[fold, 're_time (sec)'] = re_time
        results_df.loc[fold, 're_memory (MB)'] = re_memory
        results_df.to_csv(manager.N_FOLD_RESULTS_FP, index=False)
        logging.debug('done')


    # Compute cross-validated results
    for fold in range(manager.N_FOLDS):
        # Get train and test data folds
        train_index, test_index = load_split_indices(
            manager.N_FOLD_CV_SPLIT_INDICIES_FP,
            fold_index=fold,
        )
        # X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Path to neural network model for this fold
        model_file_path = manager.n_fold_model_fp(fold)

        # Load extracted rules from disk
        logging.debug(
            f'Loading extracted rules from disk for '
            f'fold {fold + 1}/{manager.N_FOLDS}...'
        )
        with open(manager.n_fold_rules_fp(fold), 'rb') as rules_file:
            rules = pickle.load(rules_file)
        logging.debug('done')

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

        # Evaluate rules
        logging.debug(
            f'Evaluating rules extracted from '
            f'fold {fold + 1}/{manager.N_FOLDS}...'
        )
        re_results = evaluate(rules, manager=manager)
        logging.debug('done')

        # Save rule extraction evaluation results
        results_df = pd.read_csv(manager.N_FOLD_RESULTS_FP)
        results_df.loc[fold, 're_acc'] = re_results['acc']
        results_df.loc[fold, 're_fid'] = re_results['fid']
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

        results_df.to_csv(manager.N_FOLD_RESULTS_FP, index=False)
        logging.info(
            f"Rule extraction for fold {fold} took a total of {re_time}s and "
            f"{re_memory} MB to obtain testing accuracy {re_results['acc']} "
            f"compared to the accuracy of the neural network "
            f"{results_df.loc[fold, 'nn_accuracy']}."
        )

