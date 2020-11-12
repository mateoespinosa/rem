"""
Main implementation of the DNN rule extraction algorithm.
"""

from tqdm import tqdm  # Loading bar for rule generation
import logging
import pandas as pd
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
    ):
        self.model = keras_model
        # We will dump intermediate activations into this path if and only
        # if it is provided
        self._activations_path = activations_path

        # Keeps in memory a map between layer ID and the activations it
        # generated when we processed the given training data
        self._activation_map = {}
        self._compute_layerwise_activations(train_data)

    def _compute_layerwise_activations(self, train_data):
        """
        Store sampled activations for each layer in CSV files
        """
        # Run the network once with the whole data, and pick up intermediate
        # activations

        feature_extractor = keras.Model(
            inputs=self.model.inputs,
            outputs=[layer.output for layer in self.model.layers]
        )
        # Run this model which will output all intermediate activations
        all_features = feature_extractor.predict(train_data)

        # And now label each intermediate activation using our
        # h_{layer}_{activation} notation
        for layer_index, (layer, activation) in enumerate(zip(
            self.model.layers,
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


def extract_rules(model, train_data, verbosity=logging.INFO):
    """
    Extracts a set of rules which imitates given the provided model using the
    algorithm described in the paper.

    :param tf.keras.Model model: The model we want to imitate using our ruleset.
    :param np.array train_data: 2D data matrix containing all the training
                                   points used to train the provided keras
                                   model.
    :param logging.verbosity verbosity: The verbosity in which we want to run
                                        this algorithm.
    :returns Ruleset: the set of rules extracted from the given model.
    """

    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations
    cache_model = ModelCache(keras_model=model, train_data=train_data)

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
                threshold=(1 / num_classes),
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
                    if not new_rules:
                        pbar.write(
                            f"[ERROR] Found an empty set of rules for "
                            f"class {output_class} and layer {hidden_layer}"
                        )
                    layer_rulesets[hidden_layer].add_rules(new_rules)
                pbar.update(1)

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