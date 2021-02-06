"""
Represents a ruleset made up of rules
"""

from enum import Enum
import numpy as np
import random
import pickle

from .rule import Rule

################################################################################
## Exposed Classes
################################################################################


class RuleScoreMechanism(Enum):
    """
    This class encapsulates the different rule score mechanisms/algorithms
    we support during rule evaluation.
    """
    # Majority algorithm: every rule has score 1
    Majority = 0
    # Accuracy algorithm: every rule has a score equal to
    # (samples covered with same label as this rule)/(total samples covered)
    Accuracy = 1
    # Augmented Hill Climbing algorithm: same as used in paper.
    HillClimb = 2
    # Confidence algorithm: every rule has a score equal to its confidence
    # level
    Confidence = 3


class Ruleset(object):
    """
    Represents a set of disjunctive rules, one with its own conclusion.
    """

    def __init__(self, rules=None, feature_names=None, output_class_names=None):
        self.rules = rules or set()
        self.feature_names = feature_names
        self.output_class_map = dict(zip(
            output_class_names or [],
            range(len(output_class_names or [])),
        ))

    def __iter__(self):
        # If we iterate over this guy, it is the same as iterating over
        # its rules
        return self.rules.__iter__()

    def __len__(self):
        # The size of our ruleset will be how many rules we have inside it
        return len(self.rules)

    def _get_named_dictionary(self, instance):
        neuron_to_value_map = {}
        for i in range(len(instance)):
            neuron_name = f'h_0_{i}'
            if self.feature_names is not None:
                neuron_name = self.feature_names[i]
            neuron_to_value_map[neuron_name] = instance[i]
        return neuron_to_value_map

    def predict(self, X, use_label_names=False):
        """
        Predicts the labels corresponding to unseen data points X given a set of
        rules.

        :param np.array X: 2D matrix of data points of which we want to obtain a
            prediction for.

        :returns np.array: 1D vector with as many entries as data points in X
            containing our predicted results.
        """
        X = np.atleast_2d(X)
        y = np.array([])

        if len(X.shape) != 2:
            raise ValueError(
                "Expected provided data to be 2D but got "
                "shape {X.shape} instead."
            )

        for instance in X:
            # Map of neuron names to values from input data
            neuron_to_value_map = self._get_named_dictionary(instance)

            # Each output class given a score based on how many rules x
            # satisfies
            class_ruleset_scores = {}
            for class_rules in self.rules:
                score = class_rules.evaluate_score(
                    neuron_to_value_map
                )
                class_ruleset_scores[class_rules] = score

            # Output class with max score decides the classification of
            # instance. If a tie happens, then choose randomly
            if len(set(class_ruleset_scores.values())) == 1:
                max_class = random.choice(list(self.rules)).conclusion
            else:
                max_class = max(
                    class_ruleset_scores,
                    key=class_ruleset_scores.get
                ).conclusion

            # Output class encoding is index out output neuron
            if not use_label_names:
                # Then turn this into its encoding
                max_class = self.output_class_map.get(max_class, max_class)
            y = np.append(y, max_class)
        return y

    def rank_rules(
        self,
        X,
        y,
        score_mechanism=RuleScoreMechanism.Majority,
        k=4,
    ):
        """
        Assigns scores to all rules in this dataset given training data (X, y)
        and using the provided score mechanism to score each rule.

        :param np.darray X: 2D np.ndarray containing the training data for our
            scoring. First dimension is the sample dimension and second
            dimension is the feature dimension.
        :param np.darray y: 1D np.ndarray containing the training labels for
            datapoints in X.
            scoring.
        :param RuleScoreMechanism score_mechanism: The mechanism to use for
            our scoring function in the rules.
        :param int k: k parameter used in the augmented HillClimbing rule
            scoring mechanism.
        """

        # We will cache a map between sample IDs and their corresponding
        # maps of neurons to values to avoid computing this at all times.
        # We only do this when we use the training data for scoring our rules.
        if score_mechanism in [
            RuleScoreMechanism.Accuracy,
            RuleScoreMechanism.HillClimb,
        ]:
            neuron_map_cache = [None] * len(X)
        for class_rule in self.rules:
            # Each run of rule extraction return a DNF rule for each output
            # class
            rule_output = class_rule.conclusion

            # Each clause in the DBF rule is considered a rule for this output
            # class
            for i, clause in enumerate(class_rule.premise):
                correct = incorrect = 0

                if score_mechanism == RuleScoreMechanism.Majority:
                    # Then every single rule will have score of 1 as we will
                    # only consider the majority class in there
                    clause.score = 1
                elif score_mechanism == RuleScoreMechanism.Confidence:
                    # We will use the confidence of this clause as its scoring
                    # function as well
                    clause.score = clause.confidence
                else:
                    # Else we will score it based on
                    # Iterate over all items in the training data
                    for sample_id, (sample, label) in enumerate(zip(X, y)):
                        # Map of neuron names to values from input data. This
                        # is the form of data a rule expects
                        if neuron_map_cache[sample_id] is None:
                            # Then we are populating our cache for the first
                            # time
                            neuron_map_cache[sample_id] = \
                                self._get_named_dictionary(sample)
                        neuron_to_value_map = neuron_map_cache[sample_id]

                        # if rule predicts the correct output class
                        if clause.evaluate(data=neuron_to_value_map):
                            if rule_output == label:
                                correct += 1
                            else:
                                incorrect += 1

                    # Compute score that we will use for this clause
                    if correct + incorrect == 0:
                        clause.score = 0
                    else:
                        if score_mechanism == RuleScoreMechanism.Accuracy:
                            # Then we only use the plain accuracy to score
                            # this method
                            clause.score = correct / (correct + incorrect)
                        elif score_mechanism == RuleScoreMechanism.HillClimb:
                            clause.score = (
                                (correct - incorrect) / (correct + incorrect)
                            )
                            clause.score += correct / (incorrect + k)
                            # Then we include the extra correction term from
                            # the length of the ruleset
                            clause.score += correct / len(clause.terms)

                        else:
                            raise ValueError(
                                f"Unsupported scoring mechanism "
                                f"{score_mechanism}"
                            )

    def add_rules(self, rules):
        self.rules = self.rules.union(rules)

    def eliminate_rules(self, percent):
        """
        Eliminates the lowest scoring `percent` of clauses for each rule in this
        ruleset.

        :param float percent: A value in [0, 1] indicating the percent of
            rules we will drop.
        """
        if percent == 0:
            # Then nothing to see here
            return

        # Otherwise, let's do a quick sanity check
        if (percent > 1) or (percent < 0):
            raise ValueError(
                f'Expected given percentage to be a real number between 0 and '
                f'1 but got {percent} instead.'
            )

        # Time to actually do the elimination
        for class_rule in self.rules:
            # For each set of rules of a given class, we will sort all all the
            # individual clauses in it by their score and drop the lowest
            # `percent` of them.

            # 1. Sort all rules based on their score
            premise = sorted(list(class_rule.premise), key=lambda x: x.score)

            # 2. Eliminate the lowest `percent` percent of the rules
            to_eliminate = int(len(premise) * percent)
            premise = premise[:-to_eliminate]

            # 3. And update this class's premise
            class_rule.premise = set(premise)

    def get_rule_premises_by_conclusion(self, conclusion):
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        """
        premises = set()
        for rule in self.rules:
            if conclusion == rule.conclusion:
                premises = premises.union(rule.premise)

        return premises

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        term_confidences = {}

        for rule in self.rules:
            for clause in rule.premise:
                clause_confidence = clause.confidence
                for term in clause.terms:
                    if term in term_confidences:
                        term_confidences[term] = max(
                            term_confidences[term],
                            clause_confidence
                        )
                    else:
                        term_confidences[term] = clause_confidence

        return term_confidences

    def get_terms_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset
        """
        terms = set()
        for rule in self.rules:
            for clause in rule.premise:
                terms = terms.union(clause.terms)
        return terms

    def __str__(self):
        ruleset_str = '\n'
        for rule in sorted(self.rules, key=str):
            ruleset_str += str(rule) + '\n'

        return ruleset_str

    def get_rule_by_conclusion(self, conclusion) -> Rule:
        for rule in self.rules:
            if conclusion == rule.conclusion:
                return rule

    def get_ruleset_conclusions(self):
        conclusions = set()
        for rule in self.rules:
            conclusions.add(rule.conclusion)
        return conclusions

    def to_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    def remove_rule(self, delete_rule):
        for rule in self.rules:
            if rule.conclusion != delete_rule.conclusion:
                continue
            rule.premise = rule.premise.difference(delete_rule.premise)

    def from_file(self, path):
        with open(path, 'rb') as f:
            deserialized = pickle.load(f)
        if isinstance(deserialized, tuple):
            deserialized = deserialized[0]
        self.rules = deserialized.rules
        self.feature_names = deserialized.feature_names
        self.output_class_map = deserialized.output_class_map
        return self

    def to_json(self, **kwargs):
        result = {}
        result["rules"] = []
        for rule in self.rules:
            result["rules"].append(rule.to_json())
        result["feature_names"] = list(self.feature_names)
        result["output_class_map"] = dict(self.output_class_map)
        return result
