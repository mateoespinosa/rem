"""
Represents a ruleset made up of rules
"""

import random
import numpy as np

from .rule import Rule
from .term import Neuron


class Ruleset(object):
    """
    Represents a set of disjunctive rules
    """

    def __init__(self, rules=None):
        self.rules = rules or set()

    def __iter__(self):
        # If we iterate over this guy, it is the same as iterating over
        # its rules
        return self.rules.__iter__()

    def __len__(self):
        # The size of our ruleset will be how many rules we have inside it
        return len(self.rules)

    def predict(self, X):
        """
        Predicts the labels corresponding to unseen data points X given a set of
        rules.

        :param np.array X: 2D matrix of data points of which we want to obtain a
                           prediction for.

        :returns np.array: 1D vector with as many entries as data points in X
                           containing our predicted results.
        """
        y = np.array([])

        for instance in X:
            # Map of Neuron objects to values from input data
            neuron_to_value_map = {
                Neuron(layer=0, index=i): instance[i]
                for i in range(len(instance))
            }

            # Each output class given a score based on how many rules x
            # satisfies
            class_ruleset_scores = {}
            for class_ruleset in self.rules:
                score = class_ruleset.evaluate_rule_by_majority_voting(
                    neuron_to_value_map
                )
                class_ruleset_scores[class_ruleset] = score

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
            y = np.append(y, max_class)
        return y

    def add_rules(self, rules):
        self.rules = self.rules.union(rules)

    def get_rule_premises_by_conclusion(self, conclusion):
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        """
        premises = set()
        for rule in self.rules:
            if conclusion == rule.get_conclusion():
                premises = premises.union(rule.get_premise())

        return premises

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        term_confidences = {}

        for rule in self.rules:
            for clause in rule.get_premise():
                clause_confidence = clause.get_confidence()
                for term in clause.get_terms():
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
            for clause in rule.get_premise():
                terms = terms.union(clause.get_terms())
        return terms

    def __str__(self):
        ruleset_str = '\n'
        for rule in self.rules:
            ruleset_str += str(rule) + '\n'

        return ruleset_str

    def get_rule_by_conclusion(self, conclusion) -> Rule:
        for rule in self.rules:
            if conclusion == rule.get_conclusion():
                return rule

    def get_ruleset_conclusions(self):
        conclusions = set()
        for rule in self.rules:
            conclusions.add(rule.get_conclusion())
        return conclusions

    def combine_external_clause(self, conjunctiveClause, conclusion):
        premises = self.get_rule_premises_by_conclusion(conclusion)
        premises.add(conjunctiveClause)

        rule = self.get_rule_by_conclusion(conclusion)
        newRule = Rule(premises, conclusion)

        if rule is not None:
            self.rules.remove(rule)

        self.rules.add(newRule)

        return self.rules

    def combine_ruleset(self, other):
        conclusions_self = self.get_ruleset_conclusions()
        conclusions_other = other.get_ruleset_conclusions()
        combined_rules = set()

        diff = conclusions_self.symmetric_difference(conclusions_other)
        intersect = conclusions_self.intersection(conclusions_other)

        for rule in self.rules.union(other.rules):
            if rule.get_conclusion() in diff:
                combined_rules.add(rule)

        for rule in self.rules:
            if rule.get_conclusion() in intersect:
                premise = other.get_rule_premises_by_conclusion(
                    rule.get_conclusion()
                )
                combined_premise = premise.union(rule.get_premise())
                combined_rule = Rule(combined_premise, rule.get_conclusion())
                combined_rules.add(combined_rule)
        return combined_rules

