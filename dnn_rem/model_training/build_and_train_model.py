"""
Build neural network models given number of nodes in each hidden layer
"""

import logging
import numpy as np
import os
import sklearn
import tensorflow as tf

################################################################################
## Metric Functions
################################################################################


class LogitAUC(tf.keras.metrics.AUC):
    """
    Custom AUC metric that operates in logit activations (i.e. does not
    require them to be positive and will pass a softmax through them before
    computing the AUC)
    """
    def __init__(self, *args, **kwargs):
        super(LogitAUC, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Simply call the parent function with the argmax of the given tensor
        y_true = tf.where(
            tf.equal(tf.reduce_max(y_true, axis=-1, keepdims=True), y_true),
            1,
            0
        )
        y_pred = tf.where(
            tf.equal(tf.reduce_max(y_pred, axis=-1, keepdims=True), y_pred),
            1,
            0
        )
        super(LogitAUC, self).update_state(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )


def majority_classifier_acc(y_true, y_pred):
    """
    Helper metric function for computing the accuracy that a majority class
    predictor would obtain with the provided labels.
    """
    distr = tf.math.reduce_sum(y_true, axis=0)
    return tf.math.reduce_max(distr) / tf.math.reduce_sum(y_true)

################################################################################
## Model Definition
################################################################################


def model_fn(
    input_features,
    layer_units,
    num_outputs,
    activation="tanh",
    optimizer=None,
    last_activation="sigmoid",
    loss_function="sigmoid_xentr",
    learning_rate=0.001,
):
    """
    Model function to construct our TF model for learning our given task.

    :param int input_features: the number of features our network consumes.
    :param List[int] layer_units: The number of units in each hidden layer.
    :param int num_outputs: The number of outputs in our network.
    :param [str | function] activation: Valid keras activation function name or
        actual function to use as activation between hidden layers.
    :param [str | tf.keras.Optimizer] optimizer: Optimizer to be used for
        training.
    :param str last_activation: valid keras activation to be used
        as our last layer's activation. For now, we only support "softmax" or
        "sigmoid".
    :param str loss_function: valid keras loss function to be used
        after our last layer. This will define the used loss. For
        now, we only support "softmax_xentr" or "sigmoid_xentr".

    :returns tf.keras.Model: Compiled model for training and evaluation.
    """

    # Input layer.
    # NOTE: it is very important to have an explicit Input layer for now rather
    # than using a Sequential model. Otherwise, we will not be able to pick
    # it up correctly during rule generation.
    input_layer = tf.keras.layers.Input(shape=(input_features,))

    # And build our intermediate dense layers
    net = input_layer
    for i, units in enumerate(layer_units):
        net = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"dense_{i}",
        )(net)

    if loss_function is None and last_activation:
        if last_activation == "sigmoid":
            loss_function = "sigmoid_xentr"
        elif last_activation == "softmax":
            loss_function = "softmax_xentr"
    if last_activation is None:
        if loss_function is not None:
            raise ValueError(
                "We were not provided with a loss function or last activation "
                "function in our model."
            )
        elif "_" in loss_function:
            last_activation = loss_function[:loss_function.find("_")]
        else:
            # Default to empty so that we can do substring search
            last_activation = ""

    # And our output layer map
    net = tf.keras.layers.Dense(
        num_outputs,
        name="output_dense",
        # If the last activation is a part of the loss, then we will go ahead
        # and merge them for numerical stability. Else, let's explicitly mark
        # it here.
        activation=(
            None if (last_activation in loss_function) else last_activation
        ),
    )(net)

    # Compile Model
    model = tf.keras.models.Model(inputs=input_layer, outputs=net)
    optimizer = optimizer or tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    if loss_function == "softmax_xentr":
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=(last_activation in loss_function),
        )
    elif loss_function == "sigmoid_xentr":
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=(last_activation in loss_function),
        )
    else:
        raise ValueError(
            f"Unsupported loss {loss_function}. We currently only support "
            "softmax_xentr and softmax_xentr"
        )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            (
                LogitAUC(name='auc', multi_label=True)
                if (last_activation in loss_function)
                else tf.keras.metrics.AUC(name='auc', multi_label=True)
            ),
            'accuracy',
            majority_classifier_acc,
        ]
    )
    return model

################################################################################
## Helper functions
################################################################################


def load_model(path):
    """
    Wrapper around tf.keras.models.load_model that includes all custom layers
    and metrics we are including in our model when serializing.

    :param str path: The path of the model checkpoint we want to load.
    :returns tf.keras.Model: Model object corresponding to loaded checkpoint.
    """
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "LogitAUC": LogitAUC,
            "majority_classifier_acc": majority_classifier_acc,
        },
    )

################################################################################
## Model Train Loop
################################################################################


def run_train_loop(
    X_train,
    y_train,
    X_test,
    y_test,
    manager,
):
    """
    Builds and train our model with the given train data. Evaluates the model
    using the given test data and returns the trained model together with some
    metrics collected at test time.

    :param np.array X_train: 2D array of training data points.
    :param np.array y_train: 1D array with as many points as X_train containing
        the training labels for each point.
    :param np.array X_test: 2D array of testing data points.
    :param np.array y_test: 1D array with as many points as X_test containing
        the testing labels for each point.
    :param ExperimentManager manager: Experiment manager for handling file
        generation during our run.

    :returns Tuple[Keras.Model, int, int, int]: A tuple containing the
        trained Keras model, test accuracy, test AUC, and majority classifier
        test accuracy for the given test data.
    """
    hyperparams = manager.HYPERPARAMS

    # Weight classes due to imbalanced dataset
    class_weights = dict(enumerate(
        sklearn.utils.class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train),
            y_train
        )
    ))

    # Make sure we use a one-hot representation for this model
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    if os.path.exists(manager.BEST_NN_INIT_FP):
        # Use best saved initialisation found earlier
        logging.debug(
            f'Training neural network with best initialisation from path: '
            f'{manager.BEST_NN_INIT_FP}'
        )
        model = load_model(manager.BEST_NN_INIT_FP)
    else:
        # Build and initialise new model
        logging.debug('Training neural network with new random initialisation')
        model = model_fn(
            input_features=X_train.shape[-1],
            layer_units=hyperparams["layer_units"],
            num_outputs=manager.DATASET_INFO.n_outputs,
            last_activation=hyperparams.get("last_activation", "softmax"),
            activation=hyperparams.get("activation", "tanh"),
            learning_rate=hyperparams.get("learning_rate", 0.001)
        )

    # If on debug mode, then let's look at the architecture of the model we
    # are about to train
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        # Then let's print our model's summary for debugging purposes
        model.summary()

    # Train Model
    model.fit(
        X_train,
        y_train,
        class_weight=class_weights,
        epochs=hyperparams.get("epochs", 1),
        batch_size=hyperparams.get("batch_size", 16),
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0
        ),
    )

    # Evaluate Accuracy of the Model
    _, nn_auc, nn_accuracy, maj_class_acc = model.evaluate(
        X_test,
        y_test,
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0
        ),
    )

    return model, nn_accuracy, nn_auc, maj_class_acc
