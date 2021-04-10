"""
Baseline implementation of an algorithm to extract rules while ignoring the
given DNN's predictions. It simply uses a vanilla decision tree learner
and extracts rules from it.
"""

import numpy as np
import logging
import pandas as pd

from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.rules.C5 import C5
from dnn_rem.rules.cart import cart_rules, random_forest_rules
from dnn_rem.rules.ruleset import Ruleset

################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    train_labels,
    verbosity=logging.INFO,
    winnow=True,
    threshold_decimals=None,
    min_cases=15,
    feature_names=None,
    output_class_names=None,
    tree_extraction_algorithm_name="C5.0",
    ccp_prune=True,
    estimators=30,
    **kwargs,
):
    """
    Extracts a set of rules using the requested tree extraction algorithm and
    IGNORES the provided model.

    :param tf.keras.Model model: The model we want to imitate using our ruleset.
    :param np.array train_data: 2D data matrix containing all the training
        points used to train the provided keras model.
    :param logging.verbosity verbosity: The verbosity in which we want to run
        this algorithm.
    :param bool winnow: whether or not to use winnowing for C5.0
    :param int threshold_decimals: how many decimal points to use for
        thresholds. If None, then no truncation is done.
    :param int, float min_cases: minimum number of cases for a split to happen
        in in the used tree extraction algorithm.
    :returns Set[Rule]: the set of rules extracted from the given model.
    """
    # C5 requires y to be a pd.Series
    y = pd.Series(train_labels)

    if isinstance(tree_extraction_algorithm_name, str):
        if tree_extraction_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
            tree_extraction_algorithm = C5
            algo_kwargs = dict(
                prior_rule_confidence=1,
                winnow=winnow,
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
            )
        elif tree_extraction_algorithm.lower() == "cart":
            tree_extraction_algorithm = cart_rules
            algo_kwargs = dict(
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
                ccp_prune=ccp_prune,
            )
        elif tree_extraction_algorithm.lower() == "random_forest":
            tree_extraction_algorithm = random_forest_rules
            algo_kwargs = dict(
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
                ccp_prune=ccp_prune,
                estimators=estimators,
            )
        else:
            raise ValueError(
                f'Unsupported tree extraction algorithm '
                f'{tree_extraction_algorithm_name}. Supported algorithms are '
                '"C5.0", "CART", and "random_forest".'
            )

    assert len(train_data) == len(y), \
        'Unequal number of data instances and predictions'

    # We can extract the number of output classes from the model itself
    num_classes = model.layers[-1].output_shape[-1]

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

    rules = tree_extraction_algorithm(
        x=train_data,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        **algo_kwargs
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
