"""
Implementation of a vanilla model's compression as
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import logging
import numpy as np
import re

from tensorflow_model_optimization.python.core.sparsity.keras import \
    pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import \
    pruning_utils
from tensorflow_model_optimization.python.core.sparsity.keras import \
    pruning_impl

################################################################################
## Reimplemented TF Model Optimization Methods with Bug Fixes/Extensions
################################################################################


def factorized_pool(
    input_tensor,
    window_shape,
    pooling_type,
    strides,
    padding,
    name=None,
):
    """
    Taken from tfmo.sparse.utils and modified so that it does not squeeze all
    degenerate dimensions at the end.
    """
    if input_tensor.get_shape().ndims != 2:
        raise ValueError('factorized_pool() accepts tensors of rank 2 only')

    [height, width] = input_tensor.get_shape()
    if name is None:
        name = 'factorized_pool'
    with tf.name_scope(name):
        input_tensor_aligned = tf.reshape(input_tensor, [1, 1, height, width])

        height_pooling = tf.nn.pool(
            input_tensor_aligned,
            window_shape=[1, window_shape[0]],
            pooling_type=pooling_type,
            strides=[1, strides[0]],
            padding=padding,
        )
        swap_height_width = tf.transpose(height_pooling, perm=[0, 1, 3, 2])

        width_pooling = tf.nn.pool(
            swap_height_width,
            window_shape=[1, window_shape[1]],
            pooling_type=pooling_type,
            strides=[1, strides[1]],
            padding=padding,
        )

    return tf.squeeze(
        tf.transpose(width_pooling, perm=[0, 1, 3, 2]),
        axis=[0, 1],
    )


class Pruning(pruning_impl.Pruning):
    """
    Implementation of magnitude-based weight pruning taken from original
    source in tfmo and modified so that it uses the corrected version of
    factorized_pool.
    """

    def _maybe_update_block_mask(self, weights):
        """
        Same as the original _maybe_update_block_mask method but it uses our
        version of factorized_pool rather than the utils one.
        """
        if self._block_size == [1, 1]:
            return self._update_mask(weights)

        # TODO(pulkitb): Check if squeeze operations should now be removed since
        # we are only accepting 2-D weights.

        squeezed_weights = tf.squeeze(weights)
        abs_weights = tf.math.abs(squeezed_weights)
        pooled_weights = factorized_pool(
            abs_weights,
            window_shape=self._block_size,
            pooling_type=self._block_pooling_type,
            strides=self._block_size,
            padding='SAME',
        )

        if pooled_weights.get_shape().ndims != 2:
            pooled_weights = tf.squeeze(pooled_weights)

        new_threshold, new_mask = self._update_mask(pooled_weights)

        updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
        sliced_mask = tf.slice(
            updated_mask,
            [0, 0],
            [squeezed_weights.get_shape()[0], squeezed_weights.get_shape()[1]]
        )
        return new_threshold, tf.reshape(sliced_mask, tf.shape(weights))


class PruneLowMagnitude(pruning_wrapper.PruneLowMagnitude):
    """
    Extended version fixing bug regarding degenerate weight dimensions after
    block is applied
    """

    def build(self, input_shape):
        super(pruning_wrapper.PruneLowMagnitude, self).build(input_shape)

        weight_vars, mask_vars, threshold_vars = [], [], []

        self.prunable_weights = self.layer.get_prunable_weights()

        # For each of the prunable weights, add mask and threshold variables
        for weight in self.prunable_weights:
            mask = self.add_variable(
                'mask',
                shape=weight.shape,
                initializer=tf.keras.initializers.get('ones'),
                dtype=weight.dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )
            threshold = self.add_variable(
                'threshold',
                shape=[],
                initializer=tf.keras.initializers.get('zeros'),
                dtype=weight.dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )

            weight_vars.append(weight)
            mask_vars.append(mask)
            threshold_vars.append(threshold)
        self.pruning_vars = list(zip(weight_vars, mask_vars, threshold_vars))

        # Add a scalar tracking the number of updates to the wrapped layer.
        self.pruning_step = self.add_variable(
            'pruning_step',
            shape=[],
            initializer=tf.keras.initializers.Constant(-1),
            dtype=tf.int64,
            trainable=False,
        )

        def training_step_fn():
            return self.pruning_step

        # Create a pruning object
        self.pruning_obj = Pruning(
            training_step_fn=training_step_fn,
            pruning_vars=self.pruning_vars,
            pruning_schedule=self.pruning_schedule,
            block_size=self.block_size,
            block_pooling_type=self.block_pooling_type,
        )


def prune_low_magnitude_layer_block(
    to_prune,
    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    block_size_fn=lambda l: (1, 1),
    block_pooling_type_fn=lambda l: 'AVG',
    ignore_fn=lambda l: False,
    **kwargs
):
    """
    Extension of TF's prune_low_magnitude that allows more custom support for
    layer-by-layer block sizes and pooling functions.
    """
    def _add_pruning_wrapper(layer, **kwargs):
        if isinstance(
            layer,
            PruneLowMagnitude,
        ) or ignore_fn(layer):
            return layer
        return PruneLowMagnitude(
            layer,
            pruning_schedule=pruning_schedule,
            block_size=block_size_fn(layer),
            block_pooling_type=block_pooling_type_fn(layer),
            **kwargs
        )

    is_sequential_or_functional = isinstance(to_prune, tf.keras.Model) and (
        isinstance(to_prune, tf.keras.Sequential) or to_prune._is_graph_network
    )

      # A subclassed model is also a subclass of tf.keras.layers.Layer.
    is_keras_layer = isinstance(to_prune, tf.keras.layers.Layer) and (
        not isinstance(to_prune, tf.keras.Model)
    )

    if is_sequential_or_functional:
        return tf.keras.models.clone_model(
            to_prune,
            input_tensors=None,
            clone_function=_add_pruning_wrapper,
        )
    elif is_keras_layer:
        params.update(kwargs)
        return _add_pruning_wrapper(to_prune, **kwargs)
    else:
        raise ValueError(
            '`prune_low_magnitude` can only prune an object of the following '
            'types: tf.keras.models.Sequential, tf.keras functional model, '
            'tf.keras.layers.Layer, list of tf.keras.layers.Layer. You passed '
            f'an object of type: {to_prune.__class__.__name__}.'
        )


################################################################################
## Helper methods
################################################################################

def insert_layer_nonseq(
    model,
    layer_select,
    new_layer_fn,
    insert_layer_name=None,
    position='replace',
):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'][layer_name] = set([layer.name])
            else:
                network_dict['input_layers_of'][layer_name].add(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'][model.layers[0].name] = model.input

    # Iterate over all layers after the input
    model_outputs = []
    previous_removed_activations = None
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [
            network_dict['new_output_tensor_of'][layer_aux]
            for layer_aux in network_dict['input_layers_of'][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if layer_select(layer):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer, previous_removed_activations = new_layer_fn(
                layer=layer,
                shape=(
                    list(map(lambda y: tuple(y.shape), x)) if isinstance(x, list)
                    else x.shape
                ),
                name=insert_layer_name,
                prev_removed=previous_removed_activations,
            )
            x = new_layer(x)
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'][layer.name] = x

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model_outputs,
    )

################################################################################
## Compression Exposed API
################################################################################


def weight_magnitude_compress(
    model,
    X_train,
    y_train,
    batch_size=32,
    pruning_epochs=10,
    target_sparsity=0.75,
    initial_sparsity=0.2,
    min_delta=1e-4,
    patience=float("inf"),
    val_data=None,
    metrics=None,
    block_size_fn=lambda l: (1, 1),
    block_pooling_type_fn=lambda l: 'AVG',
    optimizer="adam",
    remove_neurons=True,
    ignore_layer_fn=lambda l: False,
):
    logging.info(f"Running compression loop to reach target {target_sparsity}")
    pruned_model = prune_low_magnitude_layer_block(
        model,
        pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=np.ceil(len(y_train) / batch_size) * pruning_epochs,
        ),
        block_pooling_type_fn=block_pooling_type_fn,
        block_size_fn=block_size_fn,
        ignore_fn=ignore_layer_fn
    )

    # `prune_low_magnitude_layer_block` requires a recompile.
    pruned_model.compile(
        optimizer=optimizer,
        loss=model.loss,
        metrics=(metrics or []),
    )
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        pruned_model.summary()

    # Fit it using early stopping as well as other logging metrics
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    if val_data is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=("val_accuracy"),
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=True,
            verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0
            ),
            mode='max',
        ))
    pruned_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=pruning_epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=(
            1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0
        ),
    )

    resulting_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # And remove neurons from dense layers if they have no weights in them
    if remove_neurons:
        def _remove_useless_neurons_dense(
            layer,
            shape,
            prev_removed=None,
            name=None,
        ):
            kernel = layer.kernel.numpy()
            col_sums = np.sum(np.abs(kernel), axis=0)
            output_remove_mask = np.isclose(col_sums, 0)
            og_config = layer.get_config()
            # Make sure we remove all neurons whose weights are empty
            og_config["units"] -= np.sum(output_remove_mask)
            og_config["name"] = name or og_config["name"]

            # Now time to set up the initial weights of this layer to be
            # the same as the ones we are replacing but with the columns of the
            # neurons we just removed completely removed
            if prev_removed is not None:
                kernel = kernel[:, ~output_remove_mask]
                kernel = kernel[~prev_removed, :]
            else:
                kernel = kernel[:, ~output_remove_mask]
            new_weights = [kernel]
            if og_config.get("use_bias", True):
                new_weights.append(layer.bias[~output_remove_mask])
            result = tf.keras.layers.Dense.from_config(og_config)
            result.build(shape[0] if isinstance(shape, list) else shape)
            result.set_weights(new_weights)
            return result, output_remove_mask

        resulting_model = insert_layer_nonseq(
            model=resulting_model,
            layer_select=lambda l: isinstance(l, tf.keras.layers.Dense),
            new_layer_fn=_remove_useless_neurons_dense,
            position='replace',
        )

    resulting_model.compile(
        optimizer=optimizer,
        loss=model.loss,
        metrics=(metrics or []),
    )
    return resulting_model


def neuron_magnitude_compress(
    model,
    X_train,
    y_train,
    batch_size=32,
    pruning_epochs=10,
    target_sparsity=0.75,
    initial_sparsity=0.2,
    min_delta=1e-4,
    patience=float("inf"),
    val_data=None,
    metrics=None,
    block_pooling_type='AVG',
    remove_neurons=True,
    ignore_layer_fn=lambda l: False,
):
    def _dense_block_size(l, **kwargs):
        if not isinstance(l, tf.keras.layers.Dense):
            # Then we do a plain weight-level block sparsity
            return (1, 1)

        # Else let's go ahead and make the block-size be a full activation
        return (l.kernel.shape[0], 1)

    return weight_magnitude_compress(
        model,
        X_train,
        y_train,
        batch_size=batch_size,
        pruning_epochs=pruning_epochs,
        target_sparsity=target_sparsity,
        initial_sparsity=initial_sparsity,
        min_delta=min_delta,
        patience=patience,
        val_data=val_data,
        metrics=metrics,
        block_size_fn=_dense_block_size,
        block_pooling_type_fn=lambda l: block_pooling_type,
        remove_neurons=remove_neurons,
        ignore_layer_fn=ignore_layer_fn,
    )
