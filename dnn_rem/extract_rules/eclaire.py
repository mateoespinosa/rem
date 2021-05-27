"""
Implementation of ECLAIRE algorithm. This algorithm extracts intermediate rules
for each hidden layer and then performs a change of variables in all of these
rule sets by using a clause-wise level rather than at a term-wise level.
This helps the model avoiding the exponential explosion of terms that arises
from distribution term-wise clauses during substitution. It also helps reducing
the variance in the ruleset sizes while also capturing correlations between
terms when extracting a ruleset for the overall clause.
"""

from multiprocessing import Pool, Lock
from tqdm import tqdm  # Loading bar for rule generation
import dill
import logging
import numpy as np
import pandas as pd
import sklearn

from .utils import ModelCache
from dnn_rem.utils.parallelism import serialized_function_execute
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset, RuleScoreMechanism
from dnn_rem.rules.C5 import C5
from dnn_rem.logic_manipulator.substitute_rules import \
    clausewise_substitute
from dnn_rem.utils.data_handling import stratified_k_fold_split
from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.rules.cart import cart_rules, random_forest_rules


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
    initial_min_cases=None,
    num_workers=1,
    feature_names=None,
    output_class_names=None,
    trials=1,
    block_size=1,
    max_number_of_samples=None,
    min_confidence=0,
    final_algorithm_name="C5.0",
    intermediate_algorithm_name="C5.0",
    estimators=30,
    ccp_prune=True,
    regression=False,
    balance_classes=False,
    intermediate_tree_max_depth=None,
    final_tree_max_depth=None,
    ecclectic=False,
    max_intermediate_rules=float("inf"),
    intermediate_drop_percent=0,  # 0.0 for original
    rule_score_mechanism=RuleScoreMechanism.Accuracy,
    per_class_elimination=True,
    **kwargs,
):
    """
    Extracts a ruleset model that approximates the given Keras model.

    This algorithm extracts intermediate rules for each hidden layer and then
    performs a change of variables in all of these rule sets by using a
    clause-wise level rather than at a term-wise level. This helps the model
    avoiding the exponential explosion of terms that arises from distribution
    term-wise clauses during substitution. It also helps reducing the variance
    in the ruleset sizes while also capturing correlations between terms when
    extracting a ruleset for the overall clause.

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
    # First determine which rule extraction algorithm we will use in this
    # setting
    if final_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
        final_algo_call = C5
        final_algo_kwargs = dict(
            winnow=winnow_features,
            threshold_decimals=threshold_decimals,
            trials=trials,
        )
    elif final_algorithm_name.lower() == "cart":
        final_algo_call = cart_rules
        final_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            ccp_prune=ccp_prune,
            class_weight="balanced",
            max_depth=final_tree_max_depth,
        )
    elif final_algorithm_name.lower() == "random_forest":
        final_algo_call = random_forest_rules
        final_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            estimators=estimators,
            max_depth=final_tree_max_depth,
        )
    else:
        raise ValueError(
            f'Unsupported tree extraction algorithm '
            f'{final_algorithm_name}. Supported algorithms are '
            '"C5.0", "CART", and "random_forest".'
        )

    if intermediate_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
        intermediate_algo_call = C5
        intermediate_algo_kwargs = dict(
            winnow=winnow_intermediate,
            threshold_decimals=threshold_decimals,
            trials=trials,
        )
        if regression:
            raise ValueError(
                f"One can only use either CART or random_forest as an "
                f"intermediate tree construction algorithm if the task in "
                f"hand if a regression task."
            )
    elif intermediate_algorithm_name.lower() == "cart":
        intermediate_algo_call = cart_rules
        intermediate_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            ccp_prune=ccp_prune,
            regression=regression,
            max_depth=intermediate_tree_max_depth,
        )
    elif intermediate_algorithm_name.lower() == "random_forest":
        intermediate_algo_call = random_forest_rules
        intermediate_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            estimators=estimators,
            regression=regression,
            max_depth=intermediate_tree_max_depth,
        )
    else:
        raise ValueError(
            f'Unsupported tree extraction algorithm '
            f'{intermediate_algorithm_name}. Supported algorithms are '
            '"C5.0", "CART", and "random_forest".'
        )

    if isinstance(rule_score_mechanism, str):
        # Then let's turn it into its corresponding enum
        rule_score_mechanism = RuleScoreMechanism.from_string(
            rule_score_mechanism
        )

    if (
        max_intermediate_rules is not None
    ) and (
        not intermediate_drop_percent
    ) and (
        max_intermediate_rules != float("inf")
    ):
        intermediate_drop_percent = 1

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

    if regression:
        class_rule_conclusion_map = None
    else:
        # Else this is a classification task
        num_classes = model.layers[-1].output_shape[-1]
        class_rule_conclusion_map = {}
        for i in range(num_classes):
            if output_class_names is not None:
                class_rule_conclusion_map[i] = output_class_names[i]
            else:
                class_rule_conclusion_map[i] = i

    if regression:
        y_predicted = np.squeeze(model.predict(train_data), axis=-1)
    else:
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

            if pbar:
                pbar.set_description(
                    f'Extracting ruleset for block output tensor '
                    f'{block_idx + 1}/{len(input_hidden_acts)} (min_cases is '
                    f'{eff_min_cases})'
                )

            new_rules = intermediate_algo_call(
                x=activation,
                y=y_predicted,
                min_cases=eff_min_cases,
                prior_rule_confidence=1,
                rule_conclusion_map=class_rule_conclusion_map,
                **intermediate_algo_kwargs
            )
            if pbar:
                pbar.update(1)

            if min_confidence:
                real_rules = set()
                for rule in new_rules:
                    new_clauses = []
                    for clause in rule.premise:
                        if clause.confidence >= min_confidence:
                            new_clauses.append(clause)

                    if new_clauses:
                        rule.premise = set(new_clauses)
                        real_rules.add(rule)
                new_rules = real_rules
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
                    activation=cache_model.get_layer_activations(
                        layer_index=x[1],
                    ),
                    block_idx=x[0],
                    layer_idx=x[1],
                    pbar=pbar
                ),
                enumerate(input_hidden_acts),
            ))
        pbar.set_description("Done extracting intermediate rulesets")

    for (block_idx, layer_idx), rules in zip(
        enumerate(input_hidden_acts),
        intermediate_rulesets,
    ):
        new_ruleset = Ruleset(
            rules=rules,
            feature_names=[
                f'h_{layer_idx}_{i}'
                for i in range(cache_model.get_num_activations(layer_idx))
            ]
        )
        if (
            (intermediate_drop_percent) and
            (new_ruleset.num_clauses() > 1)
        ):
            # Then let's do some rule dropping for compressing our
            # generated ruleset and improving the complexity of the
            # resulting algorithm
            logging.debug(
                f"Eliminating rules for ruleset of block {block_idx} using "
                f"rule ranking mechanism {rule_score_mechanism}, drop percent "
                f"{intermediate_drop_percent}, and max number of rules "
                f"{max_intermediate_rules or float('inf')}."
            )
            new_ruleset.rank_rules(
                X=cache_model.get_layer_activations(
                    layer_index=layer_idx
                ).to_numpy(),
                y=y_predicted,
                score_mechanism=rule_score_mechanism,
                use_label_names=True,
            )
            before_elimination = new_ruleset.num_clauses()
            new_ruleset.eliminate_rules(
                percent=intermediate_drop_percent,
                per_class=per_class_elimination,
                max_num=(max_intermediate_rules or float("inf")),
            )
            after_elimination = new_ruleset.num_clauses()
            logging.debug(
                f"\tRule elimination generated a rule set with "
                f"{after_elimination} rules, removing a total of "
                f"{before_elimination - after_elimination} rules."
            )

        intermediate_rulesets[block_idx] = new_ruleset

    # Now time to replace all intermediate clauses with clauses that only
    # depend on the input activations
    if ecclectic:
        end_rules = intermediate_algo_call(
            x=cache_model.get_layer_activations(
                layer_index=0,
            ),
            y=y_predicted,
            min_cases=min_cases,
            prior_rule_confidence=1,
            rule_conclusion_map=class_rule_conclusion_map,
            **intermediate_algo_kwargs
        )
    else:
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

                target = pd.Series(
                    data=[True for _ in range(input_acts.shape[0])]
                )
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
                if balance_classes and (
                    final_algorithm_name.lower() in ["c5.0", "c5", "see5"]
                ):
                    # Then let's extend it so that it supports unbalanced
                    # cases in here
                    class_weights = \
                        sklearn.utils.class_weight.compute_class_weight(
                            'balanced',
                            np.unique(target),
                            target
                        )
                    case_weights = [1 for _ in target]
                    for i, label in enumerate(target):
                        case_weights[i] = class_weights[int(label)]

                    case_weights = pd.Series(data=case_weights)
                    final_algo_kwargs['case_weights'] = case_weights

                new_rules = final_algo_call(
                    x=input_acts,
                    y=target,
                    rule_conclusion_map={
                        True: clause,
                        False: f"not_{clause}",
                    },
                    prior_rule_confidence=clause.confidence,
                    min_cases=min_cases,
                    **final_algo_kwargs
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
                        serialized_function_execute,
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
