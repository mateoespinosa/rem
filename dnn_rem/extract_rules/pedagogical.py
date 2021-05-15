"""
Baseline implementation of an algorithm to extract rules from a DNN using a
simple pedagogical algorithm: we extract a decision tree that maps input
features with the model's outputs.
"""

import numpy as np
import logging
import pandas as pd

from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.rules.C5 import C5
from dnn_rem.rules.ruleset import Ruleset
from dnn_rem.utils.data_handling import stratified_k_fold_split
from dnn_rem.rules.cart import cart_rules, random_forest_rules


################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    train_labels=None,
    verbosity=logging.INFO,
    winnow=True,
    threshold_decimals=None,
    min_cases=15,
    feature_names=None,
    output_class_names=None,
    max_number_of_samples=None,
    tree_extraction_algorithm_name="C5.0",
    trials=1,
    tree_max_depth=None,
    ccp_prune=True,
    estimators=30,
    regression=False,
    **kwargs,
):
    """
    Extracts a set of rules which imitates given the provided model in a
    pedagogical manner using C5 on the outputs and inputs of the network.

    :param tf.keras.Model model: The model we want to imitate using our ruleset.
    :param np.array train_data: 2D data matrix containing all the training
        points used to train the provided keras model.
    :param logging.verbosity verbosity: The verbosity in which we want to run
        this algorithm.
    :param bool winnow: whether or not to use winnowing for C5.0
    :param int threshold_decimals: how many decimal points to use for
        thresholds. If None, then no truncation is done.
    :param int min_cases: minimum number of cases for a split to happen in C5.0
    :returns Set[Rule]: the set of rules extracted from the given model.
    """

    if tree_extraction_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
        algo_call = C5
        algo_kwargs = dict(
            winnow=winnow,
            threshold_decimals=threshold_decimals,
            trials=trials,
        )
    elif tree_extraction_algorithm_name.lower() == "cart":
        algo_call = cart_rules
        algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            ccp_prune=ccp_prune,
            class_weight="balanced",
            max_depth=tree_max_depth,
            regression=regression,
        )
    elif tree_extraction_algorithm_name.lower() == "random_forest":
        algo_call = random_forest_rules
        algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            estimators=estimators,
            regression=regression,
        )
    else:
        raise ValueError(
            f'Unsupported tree extraction algorithm '
            f'{tree_extraction_algorithm_name}. Supported algorithms are '
            '"C5.0", "CART", and "random_forest".'
        )

    # Determine whether we want to subsample our training dataset to make it
    # more scalable or not
    sample_fraction = 0
    if max_number_of_samples is not None:
        if max_number_of_samples < 1:
            sample_fraction = max_number_of_samples
        elif max_number_of_samples < train_data.shape[0]:
            sample_fraction = max_number_of_samples / train_data.shape[0]

    if sample_fraction and (train_labels is not None):
        [(new_indices, _)] = stratified_k_fold_split(
            X=train_data,
            y=train_labels,
            n_folds=1,
            test_size=(1 - sample_fraction),
            random_state=42,
        )
        train_data = train_data[new_indices, :]
        train_labels = train_labels[new_indices]

    # y = output classifications of neural network. C5 requires y to be a
    # pd.Series
    nn_model_predictions = model.predict(train_data)
    if regression:
        nn_model_predictions = np.squeeze(nn_model_predictions, axis=-1)
    else:
        nn_model_predictions = np.argmax(nn_model_predictions, axis=-1)
    y = pd.Series(nn_model_predictions)

    assert len(train_data) == len(y), \
        'Unequal number of data instances and predictions'

    # We can extract the number of output classes from the model itself
    num_classes = model.layers[-1].output_shape[-1]

    # Use C5 to extract rules using only input and output values of the network
    # C5 returns disjunctive rules with conjunctive terms
    train_data = pd.DataFrame(
        data=train_data,
        columns=[
            feature_names[i] if feature_names is not None else f"h_0_{i}"
            for i in range(train_data.shape[-1])
        ],
    )

    if regression:
        rule_conclusion_map = None
    else:
        rule_conclusion_map = {}
        for i in range(num_classes):
            if output_class_names is not None:
                rule_conclusion_map[i] = output_class_names[i]
            else:
                rule_conclusion_map[i] = i

    rules = algo_call(
        x=train_data,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        prior_rule_confidence=1,
        min_cases=min_cases,
        **algo_kwargs,
    )

    # Merge rules so that they are in Disjunctive Normal Form
    # Now there should be only 1 rule per rule conclusion
    # Ruleset is encapsulated/represented by a DNF rule
    # dnf_rules is a set of rules
    dnf_rules = merge(rules)

    return Ruleset(
        dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
