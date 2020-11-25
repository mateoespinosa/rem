"""
Main implementation of the DNN rule extraction algorithm.
"""

from tqdm import tqdm  # Loading bar for rule generation
import logging
import pandas as pd
import scipy.special as activation_fns
import tensorflow.keras.models as keras

from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset
from dnn_rem.rules.C5 import C5
from dnn_rem.rules.term import TermOperator
from dnn_rem.logic_manipulator.substitute_rules import substitute


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
    ):
        self._model = keras_model
        # We will dump intermediate activations into this path if and only
        # if it is provided
        self._activations_path = activations_path

        # Keeps in memory a map between layer ID and the activations it
        # generated when we processed the given training data
        self._activation_map = {}
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
            neuron_labels = [
                f'h_{layer_index}_{i}'
                for i in range(out_shape[-1])
            ]

            # For the last layer, let's make sure it is turned into a
            # probability distribution in case the operation was merged into
            # the loss function. This is needed when the last activation (
            # e.g., softmax) is merged into the loss function (
            # e.g., softmax_cross_entropy).
            if last_activation and (layer_index == (len(self) - 1)):
                if last_activation == "softmax":
                    activation = activation_fns.softmax(activation)
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

    def get_layer_activations(self, layer_index):
        """
        Return activation values given layer index
        """
        return self._activation_map[layer_index]

    def get_layer_activations_of_neuron(self, layer_index, neuron_index):
        """
        Return activation values given layer index, only return the column for
        a given neuron index
        """
        neuron_key = f'h_{layer_index}_{neuron_index}'
        return self.get_layer_activations(layer_index)[neuron_key]


################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    verbosity=logging.INFO,
    last_activation=None,
):
    """
    Extracts a set of rules which imitates given the provided model using the
    algorithm described in the paper.

    :param tf.keras.Model model: The model we want to imitate using our ruleset.
    :param np.array train_data: 2D data matrix containing all the training
        points used to train the provided keras model.
    :param logging.verbosity verbosity: The verbosity in which we want to run
        this algorithm.
    :param str last_activation: an explicit function name to apply to the
        activations of the last layer of the given model before rule extraction.
        This is needed in case the network's last activation function got merged
        into the network's loss. If None, then no activation is done. Otherwise,
        it must be either "sigmoid" or "softmax".

    :returns Ruleset: the set of rules extracted from the given model.
    """

    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations
    cache_model = ModelCache(
        keras_model=model,
        train_data=train_data,
        last_activation=last_activation,
    )

    # Now time to actually extract our set of rules
    dnf_rules = set()

    # Compute our total looping space for purposes of logging our progress
    num_classes = model.layers[-1].output_shape[-1]
    total_loop_volume = num_classes * (len(model.layers) - 1)
    with tqdm(
        total=total_loop_volume,
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        for output_class in range(num_classes):
            layer_rulesets = [Ruleset() for _ in model.layers]

            # Initial output layer rule
            output_layer = len(model.layers) - 1
            initial_rule = Rule.initial_rule(
                output_layer=output_layer,
                output_class=output_class,
                # If we use sigmoid cross-entropy loss, then this threshold
                # becomes 0.5 and does not depend on the number of classes.
                # Also if activation function is not provided, we will default
                # to using 0.5 thresholds.
                threshold=(
                    (1 / num_classes) if (last_activation == "softmax") else 0.5
                ),
            )
            layer_rulesets[output_layer].add_rules({initial_rule})

            # Extract layer-wise rules
            for hidden_layer in reversed(range(output_layer)):
                predictors = cache_model.get_layer_activations(
                    layer_index=hidden_layer,
                )

                term_confidences = layer_rulesets[
                    hidden_layer + 1
                ].get_terms_with_conf_from_rule_premises()
                terms = term_confidences.keys()
                for i, term in enumerate(terms, start=1):
                    pbar.set_description(
                        f'Extracting rules for term {i}/{len(terms)} of '
                        f'layer {hidden_layer} for class {output_class}'
                    )

                    #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                    target = term.apply(
                        cache_model.get_layer_activations_of_neuron(
                            layer_index=(hidden_layer + 1),
                            neuron_index=term.get_neuron_index()
                        )
                    )

                    prior_rule_confidence = term_confidences[term]
                    rule_conclusion_map = {
                        True: term,
                        False: term.negate(),
                    }
                    new_rules = C5(
                        x=predictors,
                        y=target,
                        rule_conclusion_map=rule_conclusion_map,
                        prior_rule_confidence=prior_rule_confidence,
                    )

                    layer_rulesets[hidden_layer].add_rules(new_rules)
                    pbar.update(1/len(terms))

                if not len(layer_rulesets[hidden_layer]):
                    pbar.write(
                        f"[WARNING] Found an empty set of rules for "
                        f"class {output_class} and layer {hidden_layer}"
                    )

            # Merge layer-wise rules
            output_rule = initial_rule
            for hidden_layer in reversed(range(output_layer)):
                pbar.set_description(
                    f"Substituting rules for layer {hidden_layer}"
                )
                output_rule = substitute(
                    total_rule=output_rule,
                    intermediate_rules=layer_rulesets[hidden_layer],
                )
            dnf_rules.add(output_rule)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(rules=dnf_rules)
