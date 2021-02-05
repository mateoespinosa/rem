"""
Methods used for evaluating the performance of a given set of rules.
"""

import sklearn
import tensorflow as tf

from . import metrics


def evaluate(
    ruleset,
    X_test,
    y_test,
    high_fidelity_predictions,
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
    predicted_labels = ruleset.predict(X_test)

    # Compute Accuracy
    acc = sklearn.metrics.accuracy_score(predicted_labels, y_test)

    # Compute Fidelity
    fid = metrics.fidelity(predicted_labels, high_fidelity_predictions)

    # Compute Comprehensibility
    comprehensibility_results = metrics.comprehensibility(ruleset)

    # Overlapping features
    n_overlapping_features = metrics.overlapping_features(ruleset)

    # Compute the AUC using this model. For multiple labels, we average
    # across all labels
    auc = sklearn.metrics.roc_auc_score(
        tf.keras.utils.to_categorical(y_test),
        tf.keras.utils.to_categorical(predicted_labels),
        multi_class="ovr",
        average='samples',
    )

    # And wrap them all together
    results = dict(
        acc=acc,
        fid=fid,
        n_overlapping_features=n_overlapping_features,
        auc=auc,
    )
    results.update(comprehensibility_results)

    return results
