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

################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    verbosity=logging.INFO,
    winnow=True,
    threshold_decimals=None,
    min_cases=15,
    feature_names=None,
    output_class_names=None,
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

    # y = output classifications of neural network. C5 requires y to be a
    # pd.Series
    nn_model_predictions = np.argmax(model.predict(train_data), axis=-1)
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

    rule_conclusion_map = {}
    for i in range(num_classes):
        if output_class_names is not None:
            rule_conclusion_map[i] = output_class_names[i]
        else:
            rule_conclusion_map[i] = i

    rules = C5(
        x=train_data,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        prior_rule_confidence=1,
        winnow=winnow,
        threshold_decimals=threshold_decimals,
        min_cases=min_cases,
    )

    # Merge rules so that they are in Disjunctive Normal Form
    # Now there should be only 1 rule per rule conclusion
    # Ruleset is encapsulated/represented by a DNF rule
    # dnf_rules is a set of rules
    dnf_rules = merge(rules)
    assert len(dnf_rules) == num_classes, \
        f'Should only exist 1 DNF rule per class: {rules} vs {dnf_rules}'

    return Ruleset(
        dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
