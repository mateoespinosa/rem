"""
Helper utilities common to several rule extractors.
"""

import logging
import pandas as pd
import scipy.special as activation_fns
import tensorflow.keras.models as keras


################################################################################
## Helper Classes
################################################################################

class ModelCache(object):
    """
    Represents trained neural network model. Used as a cache mechanism for
    storing intermediate activation values of an executed model.
    """

    def __init__(
        self,
        keras_model,
        train_data,
        activations_path=None,
        last_activation=None,
        feature_names=None,
        output_class_names=None,
    ):
        self._model = keras_model
        # We will dump intermediate activations into this path if and only
        # if it is provided
        self._activations_path = activations_path

        # Keeps in memory a map between layer ID and the activations it
        # generated when we processed the given training data
        self._activation_map = {}
        self._feature_names = feature_names
        self._output_class_names = output_class_names

        self._compute_layerwise_activations(
            train_data=train_data,
            last_activation=last_activation,
        )

    def __len__(self):
        """
        Returns the number of layers in this cache.
        """
        return len(self._model.layers)

    def _compute_layerwise_activations(self, train_data, last_activation=None):
        """
        Store sampled activations for each layer in CSV files
        """
        # Run the network once with the whole data, and pick up intermediate
        # activations

        feature_extractor = keras.Model(
            inputs=self._model.inputs,
            outputs=[layer.output for layer in self._model.layers]
        )
        # Run this model which will output all intermediate activations
        all_features = feature_extractor.predict(train_data)

        # And now label each intermediate activation using our
        # h_{layer}_{activation} notation
        for layer_index, (layer, activation) in enumerate(zip(
            self._model.layers,
            all_features,
        )):
            # e.g. h_1_0, h_1_1, ..
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                if len(out_shape) == 1:
                    # Then we will allow degenerate singleton inputs
                    [out_shape] = out_shape
                else:
                    # Else this is not a sequential model!!
                    raise ValueError(
                        f"We encountered some branding in input model with "
                        f"layer at index {layer_index}"
                    )
            neuron_labels = []
            for i in range(out_shape[-1]):
                if (layer_index == 0) and (self._feature_names is not None):
                    neuron_labels.append(self._feature_names[i])
                elif (layer_index == (len(self) - 1)) and (
                    self._output_class_names is not None
                ):
                    neuron_labels.append(self._output_class_names[i])
                else:
                    neuron_labels.append(f'h_{layer_index}_{i}')

            # For the last layer, let's make sure it is turned into a
            # probability distribution in case the operation was merged into
            # the loss function. This is needed when the last activation (
            # e.g., softmax) is merged into the loss function (
            # e.g., softmax_cross_entropy).
            if last_activation and (layer_index == (len(self) - 1)):
                if last_activation == "softmax":
                    activation = activation_fns.softmax(activation, axis=-1)
                elif last_activation == "sigmoid":
                    # Else time to use sigmoid function here instead
                    activation = activation_fns.expit(activation)
                else:
                    raise ValueError(
                        f"We do not support last activation {last_activation}"
                    )

            self._activation_map[layer_index] = pd.DataFrame(
                data=activation,
                columns=neuron_labels,
            )

            if self._activations_path is not None:
                self._activation_map[layer_index].to_csv(
                    f'{self._activations_path}{layer_index}.csv',
                    index=False,
                )
        logging.debug('Computed layerwise activations.')

    def get_layer_activations(self, layer_index, top_k=1):
        """
        Return activation values given layer index
        """
        result = self._activation_map[layer_index]
        if (top_k != 1):
            np_preds = result.to_numpy()
            top_inds = np.argsort(np.mean(np.abs(np_preds), axis=0))
            top_k = 0.1
            top_k_indices = top_inds[-int(np.ceil(len(top_inds) * top_k)):]
            result = result.iloc[:, top_k_indices]
        return result

    def get_num_activations(self, layer_index):
        """
        Return the number of activations for the layer at the given index.
        """
        return self._activation_map[layer_index].shape[-1]

    def get_layer_activations_of_neuron(self, layer_index, neuron_index):
        """
        Return activation values given layer index, only return the column for
        a given neuron index
        """
        neuron_key = f'h_{layer_index}_{neuron_index}'
        if (layer_index == 0) and self._feature_names:
            neuron_key = self._feature_names[neuron_index]
        if (layer_index == (len(self) - 1)) and self._output_class_names:
            neuron_key = self._output_class_names[neuron_index]

        return self.get_layer_activations(layer_index)[neuron_key]
