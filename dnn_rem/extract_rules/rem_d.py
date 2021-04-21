"""
Main implementation of the vanilla REM-D rule extraction algorithm for DNNs.
"""
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation
import dill
import logging
import numpy as np
import pandas as pd
import scipy.special as activation_fns
import tensorflow.keras.models as keras
from sklearn.model_selection import StratifiedShuffleSplit

from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset, RuleScoreMechanism
from dnn_rem.rules.C5 import C5
from dnn_rem.logic_manipulator.substitute_rules import \
    substitute
from dnn_rem.logic_manipulator.delete_redundant_terms import \
    global_most_general_replacement, remove_redundant_terms
from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.utils.data_handling import stratified_k_fold_split

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


def _serialized_function_execute(serialized):
    """
    Helper function to execute a serialized serialized (using dill)

    :param str serialized: The string containing a serialized tuple
        (function, args) which was generated using dill.
    :returns X: the result of executing function(*args)
    """
    function, args = dill.loads(serialized)
    return function(*args)




################################################################################
## Exposed Methods
################################################################################

def extract_rules(
    model,
    train_data,
    train_labels=None,
    verbosity=logging.INFO,
    last_activation=None,
    threshold_decimals=None,
    winnow_intermediate=True,
    winnow_features=True,
    min_cases=15,
    num_workers=1,  # 1 for original
    feature_names=None,
    output_class_names=None,
    preemptive_redundant_removal=False,  # False for original
    top_k_activations=1,  # 1 for original
    intermediate_drop_percent=0,  # 0.0 for original
    initial_drop_percent=None,  # None for original
    rule_score_mechanism=RuleScoreMechanism.Accuracy,
    trials=1,  # 1 for original
    block_size=1,  # 1 for original
    merge_repeated_terms=False,  # False for original
    max_number_of_samples=None,
    **kwargs,
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
    :param bool winnow: whether or not to use winnowing for C5.0
    :param int threshold_decimals: how many decimal points to use for
        thresholds. If None, then no truncation is done.
    :param int min_cases: minimum number of cases for a split to happen in C5.0

    :returns Ruleset: the set of rules extracted from the given model.
    """
    # Determine whether we want to subsample our training dataset to make it
    # more scalable or not
    sample_fraction = 0
    if max_number_of_samples is not None:
        if max_number_of_samples < 1:
            sample_fraction = max_number_of_samples
        elif max_number_of_samples < train_data.shape[0]:
            sample_fraction = max_number_of_samples / train_data.shape[0]

    if sample_fraction and (train_labels is not None):
        [(new_indices, _)] = stratified_k_fold_split(
            X=train_data,
            y=train_labels,
            n_folds=1,
            test_size=(1 - sample_fraction),
            random_state=42,
        )
        train_data = train_data[new_indices, :]
        train_labels = train_labels[new_indices]

    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations
    cache_model = ModelCache(
        keras_model=model,
        train_data=train_data,
        last_activation=last_activation,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )

    if initial_drop_percent is None:
        # Then we do a constant dropping rate through the entire network
        initial_drop_percent = intermediate_drop_percent

    if isinstance(rule_score_mechanism, str):
        # Then let's turn it into its corresponding enum
        rule_score_mechanism = RuleScoreMechanism.from_string(
            rule_score_mechanism
        )

    # Now time to actually extract our set of rules
    dnf_rules = set()

    # Compute our total looping space for purposes of logging our progress
    output_layer = len(model.layers) - 1
    input_hidden_acts = list(range(0, output_layer, block_size))
    output_hidden_acts = input_hidden_acts[1:] + [output_layer]

    num_classes = model.layers[-1].output_shape[-1]
    total_loop_volume = num_classes * (len(input_hidden_acts) - 1)

    with tqdm(
        total=total_loop_volume,
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        for output_class_idx in range(num_classes):
            if output_class_names:
                output_class_name = output_class_names[output_class_idx]
            else:
                output_class_name = str(output_class_idx)

            # Initial output layer rule
            class_rule = Rule.initial_rule(
                output_class=output_class_name,
                # If we use sigmoid cross-entropy loss, then this threshold
                # becomes 0.5 and does not depend on the number of classes.
                # Also if activation function is not provided, we will default
                # to using 0.5 thresholds.
                threshold=(
                    (1 / num_classes) if (last_activation == "softmax") else 0.5
                ),
            )
            # Extract layer-wise rules

            for hidden_layer, next_hidden_layer in zip(
                reversed(input_hidden_acts),
                reversed(output_hidden_acts),
            ):
                # Obtain our cached predictions
                predictors = cache_model.get_layer_activations(
                    layer_index=hidden_layer,
                    # We never prune things from the input layer itself
                    top_k=top_k_activations if hidden_layer else 1,
                )

                # We will generate an intermediate ruleset for this layer
                intermediate_rules = Ruleset(
                    feature_names=list(predictors.columns)
                )

                # And time to call C5.0 for each term
                term_confidences = \
                    class_rule.get_terms_with_conf_from_rule_premises()
                partial_terms = list(term_confidences.keys())
                # And get rid of terms that are negations of each other
                if merge_repeated_terms:
                    terms = set()
                    for term in partial_terms:
                        if term.negate() in terms:
                            # Then no need to add this guy
                            continue
                        terms.add(term)
                    terms = list(terms)
                else:
                    terms = partial_terms

                if preemptive_redundant_removal:
                    terms = remove_redundant_terms(terms)
                num_terms = len(terms)

                # We preemptively extract all the activations of the next layer
                # so that we can serialize the function below using dill.
                # Otherwise, we will hit issues due to Pandas dataframes not
                # being compatible with dill/pickle
                next_layer_activations = cache_model.get_layer_activations(
                    layer_index=next_hidden_layer,
                )

                # Helper method to extract rules from the terms coming from a
                # hidden layer and a given label. We encapsulate it as an
                # anonymous function for it to be able to be used in a
                # multi-process fashion.
                def _extract_rules_from_term(term, i=None, pbar=None):
                    if pbar and (i is not None):
                        pbar.set_description(
                            f'Extracting rules for term {i}/'
                            f'{num_terms} {term} of layer '
                            f'{hidden_layer} for class {output_class_name}'
                        )

                    #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                    target = term.apply(
                        next_layer_activations[str(term.variable)]
                    )
                    logging.debug(
                        f"\tA total of {np.count_nonzero(target)}/"
                        f"{len(target)} training samples satisfied {term}."
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
                        winnow=(
                            winnow_intermediate if hidden_layer
                            else winnow_features
                        ),
                        threshold_decimals=threshold_decimals,
                        min_cases=min_cases,
                        trials=trials,
                    )
                    if pbar:
                        pbar.update(1/num_terms)
                    return new_rules

                # Now compute the effective number of workers we've got as
                # it can be less than the provided ones if we have less terms
                effective_workers = min(num_workers, num_terms)
                if effective_workers > 1:
                    # Them time to do this the multi-process way
                    pbar.set_description(
                        f"Extracting rules for layer {hidden_layer} of with "
                        f"output class {output_class_name} using "
                        f"{effective_workers} new processes for {num_terms} "
                        f"terms"
                    )
                    with Pool(processes=effective_workers) as pool:
                        # Now time to do a multiprocess map call. Because this
                        # needs to operate only on serializable objects, what
                        # we will do is the following: we will serialize each
                        # partition bound and the function we are applying
                        # into a tuple using dill and then the map operation
                        # will deserialize each entry using dill and execute
                        # the provided method
                        serialized_terms = [None for _ in range(len(terms))]
                        for j, term in enumerate(sorted(terms, key=str)):
                            # Let's serialize our (function, args) tuple
                            serialized_terms[j] = dill.dumps(
                                (_extract_rules_from_term, (term,))
                            )

                        # And do the multi-process pooling call
                        new_rulesets = pool.map(
                            _serialized_function_execute,
                            serialized_terms,
                        )

                    # And update our bar with only one step as we do not have
                    # the granularity we do in the non-multi-process way
                    pbar.update(1)
                else:
                    # Else we will do it in this same process in one jump
                    new_rulesets = list(map(
                        lambda x: _extract_rules_from_term(
                            term=x[1],
                            i=x[0],
                            pbar=pbar
                        ),
                        enumerate(sorted(terms, key=str), start=1),
                    ))

                # Time to do our simple reduction from our map above by
                # accumulating all the generated rules into a single ruleset
                for ruleset in new_rulesets:
                    intermediate_rules.add_rules(ruleset)

                logging.debug(
                    f'\tGenerated intermediate ruleset for layer '
                    f'{hidden_layer} and output class {output_class_name} has '
                    f'{intermediate_rules.num_clauses()} rules and '
                    f'{intermediate_rules.num_terms()} different terms in it.'
                )

                # Merge rules with current accumulation
                pbar.set_description(
                    f"Substituting rules for layer {hidden_layer} with output "
                    f"class {output_class_name}"
                )
                class_rule = substitute(
                    total_rule=class_rule,
                    intermediate_rules=intermediate_rules,
                )

                if not len(class_rule.premise):
                    pbar.write(
                        f"[WARNING] Found rule with empty premise of for "
                        f"class {output_class_name}."
                    )

                # And then time to drop some intermediate rules in here!
                temp_ruleset = Ruleset(
                    rules=[class_rule],
                    feature_names=list(predictors.columns),
                    output_class_names=[output_class_name, None],
                )
                if (
                    (train_labels is not None) and
                    (intermediate_drop_percent) and
                    (temp_ruleset.num_clauses() > 1)
                ):
                    # Then let's do some rule dropping for compressing our
                    # generated ruleset and improving the complexity of the
                    # resulting algorithm



                    term_target = []
                    for label in train_labels:
                        if label == output_class_idx:
                            term_target.append(output_class_name)
                        else:
                            term_target.append(None)
                    temp_ruleset.rank_rules(
                        X=cache_model.get_layer_activations(
                            layer_index=hidden_layer
                        ).to_numpy(),
                        y=term_target,
                        score_mechanism=rule_score_mechanism,
                        use_label_names=True,
                    )

                    slope = (intermediate_drop_percent - initial_drop_percent)
                    slope = slope/(output_layer - 1)
                    eff_drop_rate = intermediate_drop_percent - (
                        slope * hidden_layer
                    )
                    temp_ruleset.eliminate_rules(eff_drop_rate)
                    class_rule = next(iter(temp_ruleset.rules))

            # Finally add this class rule to our solution ruleset
            dnf_rules.add(class_rule)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(
        rules=dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )

