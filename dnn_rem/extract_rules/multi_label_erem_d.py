"""
TODO
"""

from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation
import dill
import logging
import numpy as np
import pandas as pd

from .rem_d import ModelCache
from dnn_rem.utils.parallelism import serialized_function_execute
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset
from dnn_rem.rules.C5 import C5
from dnn_rem.logic_manipulator.substitute_rules import \
    multilabel_substitute
from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.utils.data_handling import stratified_k_fold_split
from dnn_rem.rules.cart import cart_rules


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
    intermediate_end_min_cases=None,
    initial_min_cases=None,  # None for original
    num_workers=1,  # 1 for original
    feature_names=None,
    output_class_names=None,
    trials=1,  # 1 for original
    block_size=1,  # 1 for original
    max_number_of_samples=None,  # None for original
    max_features=None,
    max_leaf_nodes=None,
    max_depth=None,
    ccp_prune=True,
    cart_min_cases=None,
    **kwargs,
):
    """
    Extracts a ruleset model that approximates the given Keras model.

    TODO

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

    cache_model = ModelCache(
        keras_model=model,
        train_data=train_data,
        last_activation=last_activation,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )

    if initial_min_cases is None:
        # Then we do a constant min cases through the entire network
        initial_min_cases = min_cases
    if intermediate_end_min_cases is None:
        intermediate_end_min_cases = min_cases

    if cart_min_cases is None:
        cart_min_cases = min_cases

    # Compute our total looping space for purposes of logging our progress
    output_layer = len(model.layers) - 1
    input_hidden_acts = list(range(1, output_layer, block_size))

    num_classes = model.layers[-1].output_shape[-1]
    class_rule_conclusion_map = {}
    for i in range(num_classes):
        if output_class_names is not None:
            class_rule_conclusion_map[i] = output_class_names[i]
        else:
            class_rule_conclusion_map[i] = i

    nn_model_predictions = np.argmax(model.predict(train_data), axis=-1)
    # C5 requires y to be a pd.Series
    y_predicted = pd.Series(nn_model_predictions)

    # First extract rulesets out of every intermediate block
    with tqdm(
        total=len(input_hidden_acts),
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        # Extract layer-wise rules
        def _extract_rules_from_layer(
            activation,
            layer_idx,
            block_idx,
            pbar=None,
        ):
            slope = (intermediate_end_min_cases - initial_min_cases)
            slope = slope/(len(input_hidden_acts) - 1)
            eff_min_cases = intermediate_end_min_cases - (
                slope * block_idx
            )
            if intermediate_end_min_cases > 1:
                # Then let's make sure we pass an int
                eff_min_cases = int(np.ceil(eff_min_cases))

            if pbar and (i is not None):
                pbar.set_description(
                    f'Extracting ruleset for block output tensor '
                    f'{block_idx + 1}/{len(input_hidden_acts)} (min_cases is '
                    f'{eff_min_cases})'
                )

            new_rules = C5(
                x=activation,
                y=y_predicted,
                rule_conclusion_map=class_rule_conclusion_map,
                prior_rule_confidence=1,
                winnow=winnow_intermediate,
                threshold_decimals=threshold_decimals,
                min_cases=eff_min_cases,
                trials=trials,
            )
            if pbar:
                pbar.update(1)
            return new_rules

        # Now compute the effective number of workers we've got as
        # it can be less than the provided ones if we have less terms
        effective_workers = min(num_workers, len(input_hidden_acts))
        if effective_workers > 1:
            # Them time to do this the multi-process way
            pbar.set_description(
                f"Extracting rules for all layers using "
                f"{effective_workers} new processes for "
                f"{len(input_hidden_acts)} activation blocks"
            )

            with Pool(processes=effective_workers) as pool:
                serialized_indices = [
                    None for _ in range(len(input_hidden_acts))
                ]
                for block_idx, layer_idx in enumerate(
                    input_hidden_acts
                ):
                    # Let's serialize our (function, args) tuple
                    serialized_indices[block_idx] = dill.dumps((
                        _extract_rules_from_layer,
                        (
                            cache_model.get_layer_activations(
                                layer_index=layer_idx
                            ),
                            layer_idx,
                            block_idx,
                        )
                    ))

                # And do the multi-process pooling call
                intermediate_rulesets = pool.map(
                    serialized_function_execute,
                    serialized_indices,
                )
            pbar.update(len(input_hidden_acts))
        else:
            # Else we will do it in this same process in one jump
            intermediate_rulesets = list(map(
                lambda x: _extract_rules_from_layer(
                    activations=cache_model.get_layer_activations(
                        layer_index=x[1],
                    ),
                    block_idx=x[0],
                    layer_idx=x[1],
                    pbar=pbar
                ),
                enumerate(input_hidden_acts),
            ))
        pbar.set_description("Done extracting intermediate rulesets")

    for block_idx, rules in enumerate(intermediate_rulesets):
        intermediate_rulesets[block_idx] = Ruleset(rules=rules)

    # Now time to replace all intermediate clauses with clauses that only
    # depend on the input activations
    end_rules = set()
    with tqdm(
        total=len(input_hidden_acts),
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        input_acts = cache_model.get_layer_activations(layer_index=0)
        for block_idx, layer_idx in enumerate(input_hidden_acts):
            # Obtain our cached predictions for this block's tensor
            block_out_activations = cache_model.get_layer_activations(
                layer_index=layer_idx,
            )

            # We will accumulate all rules extracted for this intermediate layer
            # into a ruleset that depends only on our input activations
            extracted_ruleset = Ruleset(feature_names=feature_names)
            layer_ruleset = intermediate_rulesets[block_idx]

            partial_terms = set()
            term_confidences = []
            for class_rule in layer_ruleset.rules:
                term_confidences.append(
                    class_rule.get_terms_with_conf_from_rule_premises()
                )
                partial_terms.update(list(term_confidences[-1].keys()))

            # And get rid of terms that are negations of each other
            terms = set()
            for term in partial_terms:
                if term.negate() in terms:
                    # Then no need to add this guy
                    continue
                terms.add(term)
            terms = list(terms)

            # We will treat all terms as independent labels and extract
            # rules by treating these as binary classes
            targets = None
            term_mapping = {}
            for i, term in enumerate(terms):
                #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                term_activations = term.apply(
                    block_out_activations[str(term.variable)]
                )
                term_mapping[term] = (i, True)
                term_mapping[term.negate()] = (i, False)

                if targets is None:
                    targets = np.expand_dims(term_activations, axis=-1)
                else:
                    targets = np.concatenate(
                        [
                            targets,
                            np.expand_dims(term_activations, axis=-1),
                        ],
                        axis=-1,
                    )
                logging.debug(
                    f"\tA total of {np.count_nonzero(term_activations)}/"
                    f"{len(term_activations)} training samples satisfied "
                    f"{term}."
                )

            pbar.set_description(
                f"Extracting rules for layer {layer_idx} of with "
                f"for {len(terms)} terms"
            )
            # Else we will do it in this same process in one jump
            multi_label_rules = cart_rules(
                x=input_acts,
                y=targets,
                threshold_decimals=threshold_decimals,
                min_cases=cart_min_cases,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                ccp_prune=ccp_prune,
                # estimators=estimators,
            )

            # Merge rules with current accumulation
            pbar.set_description(
                f"Substituting rules for layer {layer_idx}"
            )
            for i, intermediate_rule in enumerate(
                intermediate_rulesets[block_idx].rules
            ):
                new_rule = multilabel_substitute(
                    total_rule=intermediate_rule,
                    multi_label_rules=multi_label_rules,
                    term_mapping=term_mapping,
                )
                end_rules.add(new_rule)
            pbar.update(1)

    return Ruleset(
        rules=merge(end_rules),
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
