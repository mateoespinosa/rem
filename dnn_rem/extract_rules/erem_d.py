"""
Implementation of clause-wise REM-D algorithm. This algorithm acts in a
similar manner to vanilla REM-D but extracts rules at a clause-wise level rather
than at a term-wise level. This helps the model avoiding the exponential
explosion of terms that arises from distribution term-wise clauses during
substitution. It also helps reducing the variance in the ruleset sizes while
also capturing correlations between terms when extracting a ruleset for the
overall clause.
"""

from multiprocessing import Pool, Lock
from tqdm import tqdm  # Loading bar for rule generation
import dill
import logging
import numpy as np
import pandas as pd

from .rem_d import ModelCache, _serialized_function_execute
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset, RuleScoreMechanism
from dnn_rem.rules.C5 import C5
from dnn_rem.logic_manipulator.substitute_rules import \
    clausewise_substitute
from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.utils.data_handling import stratified_k_fold_split
from dnn_rem.logic_manipulator.merge import merge


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
    **kwargs,
):
    """
    Extracts a ruleset model that approximates the given Keras model.

    This algorithm acts in a similar manner to vanilla REM-D but extracts rules
    at a clause-wise level rather than at a term-wise level. This helps the
    model avoiding the exponential explosion of terms that arises from
    distribution term-wise clauses during substitution. It also helps reducing
    the variance in the ruleset sizes while also capturing correlations between
    terms when extracting a ruleset for the overall clause.

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
    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations

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
        activations = [None for _ in input_hidden_acts]
        for i, layer_idx in enumerate(input_hidden_acts):
            activations[i] = cache_model.get_layer_activations(
                layer_index=layer_idx
            )

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
                        (activations[block_idx], layer_idx, block_idx)
                    ))

                # And do the multi-process pooling call
                intermediate_rulesets = pool.map(
                    _serialized_function_execute,
                    serialized_indices,
                )
            pbar.update(len(input_hidden_acts))
        else:
            # Else we will do it in this same process in one jump
            intermediate_rulesets = list(map(
                lambda x: _extract_rules_from_layer(
                    activations=activations[x[0]],
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
            clauses = sorted(layer_ruleset.all_clauses(),  key=str)
            num_clauses = len(clauses)

            def _extract_rules_from_clause(clause, i, pbar=None):
                if pbar and (i is not None):
                    pbar.set_description(
                        f'Extracting ruleset for clause {i + 1}/'
                        f'{num_clauses} of layer {layer_idx + 1} for '
                        f'(min_cases is {min_cases})'
                    )

                target = [True for _ in range(input_acts.shape[0])]
                for term in clause.terms:
                    target = np.logical_and(
                        target,
                        term.apply(
                            block_out_activations[str(term.variable)]
                        )
                    )
                logging.debug(
                    f"\tA total of {np.count_nonzero(target)}/"
                    f"{len(target)} training samples satisfied {clause}."
                )
                rule_conclusion_map = {
                    True: clause,
                    False: f"not_{clause}",
                }
                new_rules = C5(
                    x=input_acts,
                    y=target,
                    rule_conclusion_map=rule_conclusion_map,
                    prior_rule_confidence=clause.confidence,
                    winnow=winnow_features,
                    threshold_decimals=threshold_decimals,
                    min_cases=min_cases,
                    trials=trials,
                )
                if pbar:
                    pbar.update(1/num_clauses)
                return new_rules

            # Now compute the effective number of workers we've got as
            # it can be less than the provided ones if we have less terms
            effective_workers = min(num_workers, num_clauses)
            if effective_workers > 1:
                # Them time to do this the multi-process way
                pbar.set_description(
                    f"Extracting rules for layer {layer_idx} using "
                    f"{effective_workers} new processes for {num_clauses} "
                    f"clauses"
                )
                with Pool(processes=effective_workers) as pool:
                    # Now time to do a multiprocess map call. Because this
                    # needs to operate only on serializable objects, what
                    # we will do is the following: we will serialize each
                    # partition bound and the function we are applying
                    # into a tuple using dill and then the map operation
                    # will deserialize each entry using dill and execute
                    # the provided method
                    serialized_clauses = [None for _ in range(len(clauses))]
                    for j, clause in enumerate(clauses):
                        # Let's serialize our (function, args) tuple
                        serialized_clauses[j] = dill.dumps(
                            (_extract_rules_from_clause, (clause, j))
                        )

                    # And do the multi-process pooling call
                    new_rulesets = pool.map(
                        _serialized_function_execute,
                        serialized_clauses,
                    )
                # And update our bar with only one step as we do not have
                # the granularity we do in the non-multi-process way
                pbar.update(1)
            else:
                # Else we will do it in this same process in one jump
                new_rulesets = list(map(
                    lambda x: _extract_rules_from_clause(
                        clause=x[1],
                        i=x[0],
                        pbar=pbar
                    ),
                    enumerate(clauses),
                ))

            # Time to do our simple reduction from our map above by
            # accumulating all the generated rules into a single ruleset
            for ruleset in new_rulesets:
                extracted_ruleset.add_rules(ruleset)

            extracted_ruleset.rules = merge(extracted_ruleset.rules)

            # Merge rules with current accumulation
            pbar.set_description(
                f"Substituting rules for layer {layer_idx}"
            )
            for i, intermediate_rule in enumerate(
                intermediate_rulesets[block_idx].rules
            ):
                new_rule = clausewise_substitute(
                    total_rule=intermediate_rule,
                    intermediate_rules=extracted_ruleset,
                )
                end_rules.add(new_rule)

    return Ruleset(
        rules=merge(end_rules),
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
