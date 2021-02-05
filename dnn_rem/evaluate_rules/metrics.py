"""
Metrics for evaluation of a given ruleset that was extracted from a network.
"""

from collections import Counter
import random
import copy
from ..rules.term import TermOperator

################################################################################
## Exposed Methods
################################################################################


def fidelity(predicted_labels, network_labels):
    """
    Evaluate fidelity of rules generated i.e. how well do they mimic the
    performance of the Neural Network.

    :param np.array predicted_labels:  The predicted labels from our rule set.
    :param np.array network_labels:    The labels as predicted by our original
        neural network.

    :returns float: How many labels were predicted in our rule set as they
        were predicted in the original NN model.
    """
    assert (len(predicted_labels) == len(network_labels)), \
        "Error: number of labels inconsistent !"

    return sum(predicted_labels == network_labels) / len(predicted_labels)


def comprehensibility(rules):
    """
    Computes a dictionary containing statistics on the lengths and composition
    of the rules provided.

    The number of rules per class is defined as the number of conjunctive
    clauses in a class' DNF.

    :param Iterable[Rule] rules: The rules whose compressibility we want to
        analyze.
    :returns Dictionary[str, object]: Returns a dictionary with statistics
        of the given set of rules.
    """
    all_ruleset_info = []

    for class_ruleset in rules:
        class_encoding = class_ruleset.conclusion

        # Number of rules in that class
        n_rules_in_class = len(class_ruleset.premise)

        #  Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = -min_n_terms
        total_n_terms = 0
        for clause in class_ruleset.premise:
            # Number of terms in the clause
            n_clause_terms = len(clause.terms)
            min_n_terms = min(n_clause_terms, min_n_terms)
            max_n_terms = max(n_clause_terms, max_n_terms)
            total_n_terms += n_clause_terms

        av_n_terms_per_rule = (
            (total_n_terms / n_rules_in_class) if n_rules_in_class else 0
        )

        class_ruleset_info = [
            class_encoding,
            n_rules_in_class,
            min_n_terms,
            max_n_terms,
            av_n_terms_per_rule,
        ]

        all_ruleset_info.append(class_ruleset_info)

    output_classes, n_rules, min_n_terms, max_n_terms, av_n_terms_per_rule = zip(
        *all_ruleset_info
    )
    return dict(
        output_classes=output_classes,
        n_rules_per_class=n_rules,
        min_n_terms=min_n_terms,
        max_n_terms=max_n_terms,
        av_n_terms_per_rule=av_n_terms_per_rule,
    )


def overlapping_features(rules, include_operand=False):
    # Return the number of overlapping features considered in output class
    # rulesets
    # TODO: If include operand: consider feature as a threshold on an input
    #       feature
    # TODO: this would require comparing 2 thresholds if they have the same sign
    #       but the value of threshold can differ

    all_features = []
    for class_rule in rules:
        class_features = set()
        for clause in class_rule.premise:
            for term in clause.terms:
                class_features.add(term.variable)
        all_features.append(class_features)

    # Intersection over features used in each rule
    return len(set.intersection(*all_features))


def random_features_in_rules(rules, features_name, n):
    """
    :param Iterable[Rule] rules: The rules from which we want to pick features
        at random.
    :param List[str]: List of features in the datatsetT
    :param int: Number of features desired at random

    :returns List[str]: Returns a list of n features appearing in the rules at
        random
    """

    # TODO: for some reason using set to guarantee uniqueness doesn't guarantee
    #  the same random features despite the seed. Double check.
    features_list = []
    for rule in rules:
        for clause in rule.premise:
            for term in clause.terms:
                neuron = term.variable
                feature_name = features_name[neuron.index]
                if feature_name not in features_list:
                    features_list.append(feature_name)
    sorted_list = sorted(features_list)
    random.seed(0)
    random_fav_features = random.choices(sorted_list, k=n)
    return random_fav_features


def features_recurrence(rules, features_name, n):
    """
    :param Iterable[Rule] rules: The rules whose top recurring features we want
        to analyze
    :param List[str]: List of features in the datatset
    :param int: Number of top recurring features desired

    :returns List[str]: Returns the list of top n features appearing in the
        rules the most
    """

    features_list = []
    for rule in rules:
        for clause in rule.premise:
            for term in clause.terms:
                neuron = term.variable
                feature_name = features_name[neuron.index]
                features_list.append(feature_name)

    cnt = Counter(features_list)
    d = {
        k: v
        for k, v in sorted(dict(cnt).items(), key=lambda item: -item[1])
    }
    for i in list(d)[0:n]:
        print(i[3:])
    return cnt


def features_recurrence_per_class(rules, features_name, n):
    """
    :param Iterable[Rule] rules:  The rules whose top features per class we want
        to analyze
    :param List[str]: List of features in the datatset
    :param int: Number of top recurring features per class desired

    :returns List[str]: Returns the list of top n features appearing in the
        rules the most for each class
    """

    features_dict = {}
    for rule in rules:
        class_features = []
        for clause in rule.premise:
            for term in clause.terms:
                neuron = term.variable
                feature_name = features_name[neuron.index]
                class_features.append(feature_name)
        features_dict[rule.conclusion.name] = class_features

    for item in features_dict:
        cnt = Counter(features_dict[item])
        features_dict[item] = sorted(
            dict(cnt).items(),
            key=lambda item: item[1],
            reversed=True,
        )[:n]

    return features_dict


def features_operator_frequency_recurrence_per_class(rules, features_name):
    """
    Computes a dictionary containing statistics on the number of operators
    for each feature in a class ruleset. For example:
        {
            'ilc': {
                'ge_a': [8, 6],
                'ge_b': [1, 3]
            },
            'idc': {
                'ge_a': [2, 1],
                'ge_c': [3, 3]
            }
        }
    says that for class ilc, feature ge_a appeared with great and less than
    sign 8 and 6 times, respectively.

    :param Iterable[Rule] rules: The rules whose features operators we want to
        analyze.
    :param List[str]: List of features in the datatset
    :returns Dictionary[str, Dictionary[str, List[int,int]]]:  Returns a
        dictionary with statistics on frequency of operators for each feature.
    """

    features_operator_frequency_dict = {}
    for rule in rules:
        class_feature_operator_frequency_dict = {}
        class_terms = []
        for clause in rule.premise:
            for term in clause.terms:
                class_terms.append(term)

        for term in class_terms:
            neuron = term.variable
            feature_name = features_name[neuron.index]
            if feature_name not in class_feature_operator_frequency_dict:
                op_list = [0, 0]
                if term.operator == TermOperator.GreaterThan:
                    op_list[0] += 1
                else:
                    op_list[1] += 1
                class_feature_operator_frequency_dict[feature_name] = op_list
            else:
                op_list = class_feature_operator_frequency_dict[feature_name]
                if term.operator == TermOperator.GreaterThan:
                    op_list[0] += 1
                else:
                    op_list[1] += 1

        features_operator_frequency_dict[rule.conclusion.name] = \
            class_feature_operator_frequency_dict
    return features_operator_frequency_dict


def top_features_operator_frequency_recurrence_per_class(
    rules,
    feature_names,
    n
):
    """
    Computes a dictionary containing statistics on the number of operators
    for n top recurring feature in a class ruleset. For example:
        {
            'ilc': {
                'ge_a': [8, 6],
                'ge_b': [1, 3]
            },
            'idc': {
                'ge_a': [2, 1],
                'ge_c': [3, 3]
            },
        }
    says that for class ilc, feature ge_a and ge_b were the top two recurring
    features. ge_a appeared with great and less than sign 8 and 6 times,
        respectively.

    :param Iterable[Rule] rules: The rules whose features operators we want to
        analyze.
    :param List[str]: List of features in the datatset
    :returns Dictionary[str, Dictionary[str, List[int,int]]]:  Returns a
        dictionary with statistics on frequency of operators for top n features.
    """
    recurrence = features_recurrence_per_class(rules, feature_names, n)
    operator_frequency = features_operator_frequency_recurrence_per_class(
        rules,
        feature_names,
    )
    operator_frequency_new = copy.deepcopy(operator_frequency)

    for key in operator_frequency:
        temp_list = recurrence[key]
        for k in operator_frequency[key]:
            if k not in temp_list:
                del operator_frequency_new[key][k]
    return operator_frequency_new


def features_recurrence_in_explanation(explanation, features_name):
    """
    :param Iterable[Rule] rules:  the set of rules (clauses in DNF) that act as
        an explanation for a prediction.
    :param List[str]: List of features in the datatset

    :returns Counter[str, int]: Returns a counter with statistics on the
        frequency of features in an explanation.

    """
    features_list = []
    for clause in explanation:
        for term in clause.terms:
            neuron = term.variable
            feature_name = features_name[neuron.index]
            features_list.append(feature_name)

    cnt = Counter(features_list)
    return cnt
