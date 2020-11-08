"""
Runner of our DNN rule extraction algorithm. Collects statistics to be
reported to caller.
"""

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import logging
import memory_profiler
import time


def run(
    X_train,
    y_train,
    X_test,
    y_test,
    manager,
    model_file_path,
    verbosity=logging.INFO,
):
    """
    Runs the rule extraction algorithm specified by the provided experiment
    manager the trained

    :param np.array X_train: 2D array with our training datapoints.
    :param np.array y_train: 1D array with our training labels. As many entries
                             as data points in X_train.
    :param np.array X_test: 2D array with our testing datapoints.
    :param np.array y_test: 1D array with our testing labels. As many entries
                            as data points in X_test.
    :param ExperimentManager manager: The manager used to handle data reading
                                      and dumping for our experiment.
    :param str model_file_path: A valid path to a pretrained Keras model which
                                we will use to extract rules from.
    :param logging.verbosity verbosity: The verbosity level we will use for our
                                        run.

    :returns Tuple[float, float, float, float]: indicating the following
        statistics from our rule extraction: neural network testing accuracy,
        extracted rule set, total time (in seconds) of extracting rules, and
        memory consumption (in MB)
    """

    # Make sure the data is scaled up
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Now time to load up our model
    model = load_model(model_file_path)
    _, _, nn_accuracy = model.evaluate(
        X_test,
        to_categorical(y_test),
        verbose=(
            1 if verbosity == logging.DEBUG else 0
        ),
    )

    # Rule Extraction
    start_time, start_memory = time.time(), memory_profiler.memory_usage()[0]
    rules = manager.RULE_EXTRACTOR.run(
        model=model,
        train_data=X_train,
        verbosity=verbosity,
    )
    end_time, end_memory = time.time(), memory_profiler.memory_usage()[0]

    # Rule extraction time and memory usage
    total_time = end_time - start_time
    memory = end_memory - start_memory

    return nn_accuracy, rules, total_time, memory
