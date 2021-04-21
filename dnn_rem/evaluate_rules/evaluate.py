"""
Methods used for evaluating the performance of a given set of rules.
"""

import sklearn
import logging
import numpy as np

from . import metrics

_MAX_CLAUSES = 55000

def evaluate(
    ruleset,
    X_test,
    y_test,
    high_fidelity_predictions=None,
):
    """
    Evaluates the performance of the given set of rules given the provided
    test dataset. It compares this results to a higher fidelity prediction
    of these labels (e.g. coming from a neural network)

    Will generate a dictionary with several statistics describing the nature
    and performance of the given ruleset.

    :param Ruleset ruleset: The set of rules we want to evaluate.
    :param np.ndarray X_test: testing data set for evaluation.
    :param np.ndarray y_test: testing labels of X_test for evaluation.
    :param np.ndarray high_fidelity_predictions: labels predicted for X_test
        using a high fidelity method that is not our ruleset.

    :returns Dict[str, object]: A dictionary containing several statistics and
        metrics of the current run.
    """

    # Make our predictions using our ruleset
    if ruleset.num_clauses() >= _MAX_CLAUSES:
        logging.warning(
            f"Ruleset has {ruleset.num_clauses()} clauses in it. Too many "
            f"for making an efficient prediction."
        )
        acc = 0
        auc = 0
        fid = 0
    else:
        predicted_labels = ruleset.predict(X_test)

        # Compute Accuracy
        acc = sklearn.metrics.accuracy_score(predicted_labels, y_test)

        # Compute the AUC using this model. For multiple labels, we average
        # across all labels
        auc = sklearn.metrics.roc_auc_score(
            y_test,
            predicted_labels,
            multi_class="ovr",
            average='samples',
        )

        # Compute Fidelity
        if high_fidelity_predictions is not None:
            fid = metrics.fidelity(predicted_labels, high_fidelity_predictions)
        else:
            fid = None

    # Compute Comprehensibility
    comprehensibility_results = metrics.comprehensibility(ruleset)

    # Overlapping features
    n_overlapping_features = metrics.overlapping_features(ruleset)

    # And wrap them all together
    results = dict(
        acc=acc,
        fid=fid,
        n_overlapping_features=n_overlapping_features,
        auc=auc,
    )
    results.update(comprehensibility_results)

    return results

def evaluate_estimator(
    estimator,
    X_test,
    y_test,
    high_fidelity_predictions=None,
):
    """
    Evaluates the performance of the given decision tree given the provided
    test dataset. It compares this results to a higher fidelity prediction
    of these labels (e.g. coming from a neural network)

    Will generate a dictionary with several statistics describing the nature
    and performance of the given estimator.

    :param sklearn.Estimator estimator: The set of rules we want to evaluate.
    :param np.ndarray X_test: testing data set for evaluation.
    :param np.ndarray y_test: testing labels of X_test for evaluation.
    :param np.ndarray high_fidelity_predictions: labels predicted for X_test
        using a high fidelity method that is not our estimator.

    :returns Dict[str, object]: A dictionary containing several statistics and
        metrics of the current run.
    """

    # Make our predictions using our estimator
    predicted_labels = estimator.predict(X_test)

    # Compute Accuracy
    acc = sklearn.metrics.accuracy_score(predicted_labels, y_test)

    # Compute the AUC using this model. For multiple labels, we average
    # across all labels
    auc = sklearn.metrics.roc_auc_score(
        y_test,
        predicted_labels,
        multi_class="ovr",
        average='samples',
    )

    # Compute Fidelity
    if high_fidelity_predictions is not None:
        fid = metrics.fidelity(predicted_labels, high_fidelity_predictions)
    else:
        fid = None

    # And wrap them all together
    results = dict(
        output_classes=np.unique(y_test),
        acc=acc,
        fid=fid,
        auc=auc,
    )
    return results

