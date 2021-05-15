"""
Implementation of clause-wise REM-D algorithm. This algorithm acts in a
similar manner to vanilla REM-D but extracts rules at a clause-wise level rather
than at a term-wise level. This helps the model avoiding the exponential
explosion of terms that arises from distribution term-wise clauses during
substitution. It also helps reducing the variance in the ruleset sizes while
also capturing correlations between terms when extracting a ruleset for the
overall clause.
"""

from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation
import dill
import logging
import numpy as np

from dnn_rem.rules.ruleset import Ruleset
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.C5 import C5
from dnn_rem.logic_manipulator.substitute_rules import \
    substitute

from .rem_d import ModelCache
from dnn_rem.utils.parallelism import serialized_function_execute


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
    trials=1,  # 1 for original
    block_size=1,  # 1 for original
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
            layer_rulesets = [None for _ in range(len(input_hidden_acts) + 1)]
            if output_class_names:
                output_class_name = output_class_names[output_class_idx]
            else:
                output_class_name = str(output_class_idx)

            # Initial output layer rule
            layer_rulesets[-1] = Ruleset({Rule.initial_rule(
                output_class=output_class_name,
                # If we use sigmoid cross-entropy loss, then this threshold
                # becomes 0.5 and does not depend on the number of classes.
                # Also if activation function is not provided, we will default
                # to using 0.5 thresholds.
                threshold=(
                    (1 / num_classes) if (last_activation == "softmax") else 0.5
                ),
            )})

            # Extract layer-wise rules
            prev_ruleset = layer_rulesets[-1]
            for ruleset_idx, hidden_layer, next_hidden_layer in zip(
                reversed(range(len(layer_rulesets) - 1)),
                reversed(input_hidden_acts),
                reversed(output_hidden_acts),
            ):
                # Obtain our cached predictions
                predictors = cache_model.get_layer_activations(
                    layer_index=hidden_layer,
                )

                # We will generate an intermediate ruleset for this layer
                layer_rulesets[ruleset_idx] = Ruleset(
                    feature_names=list(predictors.columns)
                )

                # And time to call C5.0 for each term
                term_confidences = layer_rulesets[
                    ruleset_idx + 1
                ].get_terms_with_conf_from_rule_premises()
                terms = list(term_confidences.keys())
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
                            serialized_function_execute,
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
                    layer_rulesets[ruleset_idx].add_rules(ruleset)

                logging.debug(
                    f'\tGenerated intermediate ruleset for layer '
                    f'{hidden_layer} and output class {output_class_name} has '
                    f'{layer_rulesets[ruleset_idx].num_clauses()} rules in it.'
                )

            # Merge rules with current accumulation
            pbar.set_description(
                f"Substituting rules for class {output_class_name}"
            )

            class_rule = None
            for i, intermediate_rules in enumerate(reversed(layer_rulesets)):
                if class_rule is None:
                    assert len(intermediate_rules.rules) == 1, \
                        len(intermediate_rules.rules)
                    class_rule = next(iter(intermediate_rules.rules))
                else:
                    pbar.set_description(
                        f"Substituting rules for layer "
                        f"{len(layer_rulesets) - i - 1} of class "
                        f"{output_class_name}."
                    )
                    class_rule = substitute(
                        total_rule=class_rule,
                        intermediate_rules=intermediate_rules,
                    )

            # Finally add this class rule to our solution ruleset
            dnf_rules.add(class_rule)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(
        rules=dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
