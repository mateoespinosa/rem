"""
Implementation of conditional-REM-D (cREM-D) where we extract clauses
for both individual terms (as in REM-D) as well as rulesets that are conditioned
on the result of all other clauses but the target one. These are then used
to create a resulting clause ruleset that takes into account correlations
between terms by using the conditioned rulesets.
"""


import dill
import logging
import numpy as np

from collections import defaultdict
from dnn_rem.logic_manipulator.merge import merge
from dnn_rem.logic_manipulator.substitute_rules import conditional_substitute
from dnn_rem.rules.C5 import C5
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset, RuleScoreMechanism
from dnn_rem.utils.parallelism import serialized_function_execute
from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation

from .utils import ModelCache


################################################################################
## Exposed Methods
################################################################################

def extract_rules(
    model,
    train_data,
    verbosity=logging.INFO,
    last_activation=None,
    threshold_decimals=None,
    winnow_intermediate=True,
    winnow_features=True,
    min_cases=15,
    num_workers=1,  # 1 for original
    feature_names=None,
    output_class_names=None,
    trials=1,  # 1 for original
    block_size=1,  # 1 for original
    **kwargs,
):
    """
    Extracts a ruleset model that approximates the given Keras model.

    To achieve this, we extract clauses for both individual terms (as in REM-D)
    as well as rulesets that are conditioned on the result of all other clauses
    but the target one. These are then used to create a resulting clause ruleset
    that takes into account correlations between terms by using the conditioned
    rulesets.

    :param keras.Model model: An input instantiated Keras Model object from
        which we will extract rules from.
    :param np.ndarray train_data: A tensor of shape [N, m] with N training
        samples which have m features each.
    :param logging.VerbosityLevel verbosity: The verbosity level to use for this
        function.
    :param str last_activation: Either "softmax" or "sigmoid" indicating which
        activation function should be applied to the last layer of the given
        model if last function is fused with loss. If None, then no activation
        function is applied.
    :param int threshold_decimals: The maximum number of decimals a threshold in
        the generated ruleset may have. If None, then we impose no limit.
    :param bool winnow_intermediate: Whether or not we use winnowing when using
        C5.0 for intermediate hidden layers.
    :param bool winnow_features: Whether or not we use winnowing when extracting
        rules in the features layer.
    :param int min_cases: The minimum number of samples we must have to perform
        a split in a decision tree.
    :param int initial_min_cases: Initial minimum number of samples required for
        a split when calling C5.0 for intermediate hidden layers. This
        value will be linearly annealed given so that the last hidden layer uses
        `initial_min_cases` and the features layer uses `min_cases`.
        If None, then it defaults to min_cases.
    :param int num_workers: Maximum number of working processes to be spanned
        when extracting rules.
    :param List[str] feature_names: List of feature names to be used for
        generating our rule set. If None, then we will assume all input features
        are named `h_0_0`, `h_0_1`, `h_0_2`, etc.
    :param List[str] output_class_names: List of output class names to be used
        for generating our rule set. If None, then we will assume all output
        are named `h_{d+1}_0`, `h_{d+1}_1`, `h_{d+1}_2`, etc where `d` is the
        number of hidden layers in the network.
    :param int trials: The number of sampling trials to use when using bagging
        for C5.0 rule extraction.
    :param int block_size: The hidden layer sampling frequency. That is, how
        often will we use a hidden layer in the input network to extract an
        intermediate rule set from it.
    :param Dict[str, Any] kwargs: The keywords arguments used for easier
        integration with other rule extraction methods.

    :returns Ruleset: the set of rules extracted from the given model.
    """
    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations
    cache_model = ModelCache(
        keras_model=model,
        train_data=train_data,
        last_activation=last_activation,
        feature_names=feature_names,
        output_class_names=output_class_names,
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
                )

                # We will generate an intermediate ruleset for this layer
                intermediate_rules = Ruleset(
                )
                indep_intermediate_rules = Ruleset(
                    feature_names=list(predictors.columns)
                )

                # And time to call C5.0 for each term
                term_confidences = \
                    class_rule.get_terms_with_conf_from_rule_premises()
                partial_terms = list(term_confidences.keys())
                # And get rid of terms that are negations of each other
                terms = set()
                for term in partial_terms:
                    if term.negate() in terms:
                        # Then no need to add this guy
                        continue
                    terms.add(term)
                terms = list(terms)
                terms = partial_terms
                terms.sort(key=str)
                term_mapping = {}
                extra_feature_names = set()
                for i, term in enumerate(terms):
                    var_name = f"__term_{i}_active"
                    term_mapping[term] = (var_name, True)
                    term_mapping[term.negate()] = (var_name, False)
                    extra_feature_names.add(var_name)

                num_terms = len(terms)

                # Construct a graph where each node is a term and there is an
                # edge between two nodes if they are being used together in the
                # same clause
                term_to_id = {}  # maps term to id
                for i, term in enumerate(terms):
                    term_to_id[term] = i
                neighbor_terms = defaultdict(set)  # Maps term id to set of
                                                   # neighbor term ids
                for clause in class_rule.premise:
                    clause_terms = list(clause.terms)
                    for i, first_term in enumerate(clause_terms):
                        id_first = term_to_id[first_term]
                        for j in range(i + 1, len(clause_terms)):
                            second_term = clause_terms[j]
                            id_second = term_to_id[second_term]
                            neighbor_terms[id_first].add(id_second)
                            neighbor_terms[id_second].add(id_first)

                # We preemptively extract all the activations of the next layer
                # so that we can serialize the function below using dill.
                # Otherwise, we will hit issues due to Pandas dataframes not
                # being compatible with dill/pickle
                next_layer_activations = cache_model.get_layer_activations(
                    layer_index=next_hidden_layer,
                )
                term_targets = [
                    term.apply(
                        next_layer_activations[str(term.variable)]
                    ) for term in terms
                ]

                # Helper method to extract rules from the terms coming from a
                # hidden layer and a given label. We encapsulate it as an
                # anonymous function for it to be able to be used in a
                # multi-process fashion.
                def _extract_rules_from_term(term, i, pbar=None):
                    if pbar and (i is not None):
                        pbar.set_description(
                            f'Extracting rules for term {i + 1}/'
                            f'{num_terms} {term} of layer '
                            f'{hidden_layer} for class {output_class_name}'
                        )

                    #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                    target = term_targets[i]
                    logging.debug(
                        f"\tA total of {np.count_nonzero(target)}/"
                        f"{len(target)} training samples satisfied {term}."
                    )

                    prior_rule_confidence = term_confidences[term]
                    rule_conclusion_map = {
                        True: term,
                        False: term.negate(),
                    }
                    # We need to extend the predictors with the labels
                    # as generated by previous terms in the ordered list
                    if num_terms > 1:
                        extended_predictors = predictors.copy()
                    else:
                        extended_predictors = predictors

                    for j in neighbor_terms[i]:
                        extended_predictors.insert(
                            loc=len(extended_predictors.columns),
                            column=f"__term_{j}_active",
                            value=list(map(int, term_targets[j])),
                            allow_duplicates=False,
                        )

                    new_cond_rules = C5(
                        x=extended_predictors,
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

                    new_indep_rules = C5(
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
                    return new_indep_rules, new_cond_rules

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
                        for j, term in enumerate(terms):
                            # Let's serialize our (function, args) tuple
                            serialized_terms[j] = dill.dumps(
                                (_extract_rules_from_term, (term, j))
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
                        enumerate(terms),
                    ))

                # Time to do our simple reduction from our map above by
                # accumulating all the generated rules into a single ruleset
                for indep_ruleset, cond_ruleset in new_rulesets:
                    intermediate_rules.add_rules(cond_ruleset)
                    indep_intermediate_rules.add_rules(indep_ruleset)

                intermediate_rules.rules = merge(intermediate_rules.rules)
                logging.debug(
                    f'\tGenerated intermediate ruleset for layer '
                    f'{hidden_layer} and output class {output_class_name} has '
                    f'{intermediate_rules.num_clauses()} rules in it.'
                )

                # Merge rules with current accumulation
                pbar.set_description(
                    f"Substituting rules for layer {hidden_layer} with output "
                    f"class {output_class_name}"
                )
                class_rule = conditional_substitute(
                    total_rule=class_rule,
                    intermediate_rules=intermediate_rules,
                    independent_intermediate_rules=indep_intermediate_rules,
                    term_mapping=term_mapping,
                    extra_feature_names=extra_feature_names,
                )
                if not len(class_rule.premise):
                    pbar.write(
                        f"[WARNING] Found rule with empty premise of for "
                        f"class {output_class_name}."
                    )

            # Finally add this class rule to our solution ruleset
            dnf_rules.add(class_rule)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(
        rules=dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
