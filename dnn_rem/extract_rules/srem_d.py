"""
Implementation of sREM-D (stands for scalable REM-D). This uses a simple
multi-label tree learning approach to capture possible correlations of
learning several decision trees for a given set of possibly related term
clauses.
It also avoids the need of distributative-substitution (as that in vanilla
REM-D) by instead substituting entire clauses with rulesets rather than
individual terms in a clause with rulesets.
"""

from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation
import logging
import numpy as np

from .utils import ModelCache
from dnn_rem.rules.rule import Rule
from dnn_rem.rules.ruleset import Ruleset, RuleScoreMechanism
from dnn_rem.rules.cart import cart_rules, random_forest_rules
from dnn_rem.logic_manipulator.substitute_rules import multilabel_substitute
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
    min_cases=15,
    feature_names=None,
    output_class_names=None,
    top_k_activations=1,  # 1 for original
    intermediate_drop_percent=0,  # 0.0 for original
    initial_drop_percent=None,  # None for original
    rule_score_mechanism=RuleScoreMechanism.Accuracy,
    block_size=1,  # 1 for original
    max_features=None,
    max_leaf_nodes=None,
    max_depth=None,
    ccp_prune=True,
    **kwargs,
):
    """
    Extracts rules by treating all terms as binary labels and extracting a
    single tree that is capable of predicting all of the given terms at once
    rather than one tree per term. When substituting, we then pick all the
    branches that lead to a conclusion where all the terms of interest in a
    given clause are activated as needed.

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
    total_loop_volume = (len(input_hidden_acts) - 1)

    with tqdm(
        total=total_loop_volume,
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        class_rules = []
        for output_class_idx in range(num_classes):
            if output_class_names:
                output_class_name = output_class_names[output_class_idx]
            else:
                output_class_name = str(output_class_idx)

            # Initial output layer rule
            class_rules.append(Rule.initial_rule(
                output_class=output_class_name,
                # If we use sigmoid cross-entropy loss, then this threshold
                # becomes 0.5 and does not depend on the number of classes.
                # Also if activation function is not provided, we will default
                # to using 0.5 thresholds.
                threshold=(
                    (1 / num_classes) if (last_activation == "softmax") else 0.5
                ),
            ))
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

            # Let's get the current terms in our class ruleset as those
            # will become target labels for our intermediate ruleset
            # extraction

            partial_terms = set()
            term_confidences = []
            for class_rule in class_rules:
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

            # We preemptively extract all the activations of the next layer
            # so that we can serialize the function below using dill.
            # Otherwise, we will hit issues due to Pandas dataframes not
            # being compatible with dill/pickle
            next_layer_activations = cache_model.get_layer_activations(
                layer_index=next_hidden_layer,
            )

            # We will treat all terms as independent labels and extract
            # rules by treating these as binary classes
            targets = None
            term_mapping = {}
            for i, term in enumerate(terms):
                #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                term_activations = term.apply(
                    next_layer_activations[str(term.variable)]
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
                f"Extracting rules for layer {hidden_layer} of with "
                f"output class {output_class_name} for {len(terms)} "
                f"terms"
            )
            # Else we will do it in this same process in one jump
            multi_label_rules = cart_rules(
                x=predictors,
                y=targets,
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                ccp_prune=ccp_prune,
            )

            # Merge rules with current accumulation
            for i, class_rule in enumerate(class_rules):
                pbar.set_description(
                    f"Substituting rules for layer {hidden_layer} with "
                    "output class "
                    f"{output_class_names[i] if output_class_names else i}"
                )
                class_rules[i] = multilabel_substitute(
                    total_rule=class_rule,
                    multi_label_rules=multi_label_rules,
                    term_mapping=term_mapping,
                )

            pbar.update(1)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(
        rules=class_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
