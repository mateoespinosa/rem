"""
Build neural network models given number of nodes in each hidden layer
"""
import logging
import numpy as np
import os
import sklearn
import tensorflow as tf


def model_fn(
    input_features,
    layer_units,
    num_outputs,
    activation="tanh",
    optimizer=None
):
    # Input layer
    dense_layers = [
        tf.keras.layers.Dense(
            units,
            input_shape=(input_features,) if i == 0 else (None,),
            activation=activation,
            name=f"dense_{i}",
        ) for i, units in enumerate(layer_units)
    ]
    model = tf.keras.Sequential(dense_layers + [
        # Output Layer
        tf.keras.layers.Dense(
            num_outputs,
            # It is crucial that the last activation of this layer is set
            # to sigmoid even though this makes it less stable when training
            # (rather than using from_logits=False in the loss).
            # The reason why this is needed is because when we iterate over our
            # algorithm, we assume the last layer has a valid probability
            # distribution
            activation="sigmoid",
            input_shape=(input_features,) if not layer_units else (None,),
            name="output_dense",
        ),
    ])

    # Compile Model
    optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(),
            'accuracy',
        ]
    )
    return model


def build_and_train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    manager,
    model_file_path,
    with_best_initilisation=False,
):
    """

    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        model_file_path: path to store trained nn model
        with_best_initilisation: if true, use initialisation saved as
            best_initialisation.h5

    Returns:
        model_accuracy: accuracy of nn model
        nn_predictions: predictions made by nn used for rule extraction
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

    if with_best_initilisation:
        # Use best saved initialisation found earlier
        logging.debug(
            f'Training neural network with best initialisation from path: '
            f'{manager.BEST_NN_INIT_FP}'
        )
        model = tf.keras.models.load_model(manager.BEST_NN_INIT_FP)
    else:
        # Build and initialise new model
        logging.debug('Training neural network with new random initialisation')
        model = model_fn(
            input_features=X_train.shape[-1],
            layer_units=hyperparams["layer_units"],
            num_outputs=manager.DATASET_INFO.n_outputs,
        )
        model.save(os.path.join(manager.TEMP_DIR, 'initialisation.h5'))

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
    _, nn_auc, nn_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0
        ),
    )

    # Save Trained Model
    model.save(model_file_path)

    return nn_accuracy, nn_auc
