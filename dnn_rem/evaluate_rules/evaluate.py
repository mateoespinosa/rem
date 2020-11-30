"""
Methods used for evaluating the performance of a given set of rules.
"""

import sklearn

from . import metrics


def evaluate(rules, X_test, y_test, high_fidelity_predictions, num_labels=2):
    """
    Evaluates the performance of the given set of rules given the provided
    test dataset. It compares this results to a higher fidelity prediction
    of these labels (e.g. coming from a neural network)

    Will generate a dictionary with several statistics describing the nature
    and performance of the given ruleset.

    :param Ruleset rules: The set of rules we want to evaluate.
    :param np.ndarray X_test: testing data set for evaluation.
    :param np.ndarray y_test: testing labels of X_test for evaluation.
    :param np.ndarray high_fidelity_predictions: labels predicted for X_test
        using a high fidelity method that is not our ruleset.

    :returns Dict[str, object]: A dictionary containing several statistics and
                                metrics of the current run.
    """

    # Make our predictions using our ruleset
    predicted_labels = rules.predict(X_test)

    # Compute Accuracy
    acc = sklearn.metrics.accuracy_score(predicted_labels, y_test)

    # Compute Fidelity
    fid = metrics.fidelity(predicted_labels, high_fidelity_predictions)

    # Compute Comprehensibility
    comprehensibility_results = metrics.comprehensibility(rules)

    # Overlapping features
    n_overlapping_features = metrics.overlapping_features(rules)

    # Compute the AUC using this model. For multiple labels, we average
    # across all labels
    avg_auc = 0
    for i in range(num_labels):
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_test == i,
            predicted_labels == i,
        )
        avg_auc += sklearn.metrics.auc(fpr, tpr)
    avg_auc /= num_labels

    # And wrap them all together
    results = dict(
        acc=acc,
        fid=fid,
        n_overlapping_features=n_overlapping_features,
        auc=avg_auc,
    )
    results.update(comprehensibility_results)

    return results
