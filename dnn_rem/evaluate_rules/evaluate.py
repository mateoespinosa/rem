"""
Methods used for evaluating the performance of a given set of rules.
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from . import metrics


def evaluate(rules, manager):
    """
    Evaluates the performance of the given set of rules given an experiment
    manager that indicates where to find the True labels and the labels
    predicted by this set of rules (as well as the ones computed by its
    corresponding NN model).

    Will generate a dictionary with several statistics describing the nature
    and performance of the given ruleset.

    :param Iterable[Rule] rules: The set of rules we want to evaluate.
    :param ExperimentManager manager: The manager in charge of organizing the
                                      data in the current experiment run.

    :returns Dict[str, object]: A dictionary containing several statistics and
                                metrics of the current run.
    """
    labels_df = pd.read_csv(manager.LABEL_FP)

    predicted_labels = labels_df[f'rule_{manager.RULE_EXTRACTOR.mode}_labels']
    true_labels = labels_df['true_labels']
    nn_labels = labels_df['nn_labels']

    # Compute Accuracy
    acc = accuracy_score(predicted_labels, true_labels)

    # Compute Fidelity
    fid = metrics.fidelity(predicted_labels, nn_labels)

    # Compute Comprehensibility
    comprehensibility_results = metrics.comprehensibility(rules)

    n_overlapping_features = metrics.overlapping_features(rules)

    results = dict(
        acc=acc,
        fid=fid,
        n_overlapping_features=n_overlapping_features,
    )
    results.update(comprehensibility_results)

    return results
