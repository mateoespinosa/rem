"""
Represent a rule with a premise in Disjunctive Normal Form (DNF) and conclusion
of another term or class conclusion
"""

from collections import defaultdict

from .clause import ConjunctiveClause
from .term import Term, Neuron
from dnn_rem.logic_manipulator.satisfiability import \
    remove_unsatisfiable_clauses


class OutputClass(object):
    """
    Represents the conclusion of a given rule. Immutable and Hashable.

    Each output class has a name and its relevant encoding in the network
    i.e. which output neuron it corresponds to
    """

    def __init__(self, name: str, encoding: int):
        self.name = name
        self.encoding = encoding

    def __str__(self):
        return f'OUTPUT_CLASS={self.name} (Neuron {self.encoding})'

    def __eq__(self, other):
        return (
            isinstance(other, OutputClass) and
            (self.name == other.name) and
            (self.encoding == other.encoding)
        )

    def __hash__(self):
        return hash((self.name, self.encoding))


class Rule(object):
    """
    Represents a rule in DNF form i.e.
        (t1 AND t2 AND ..) OR ... OR  ( tk AND tk2 AND ... )  -> <conclusion>

    Immutable and Hashable.
    """

    def __init__(self, premise, conclusion):
        self.premise = remove_unsatisfiable_clauses(clauses=premise)
        self.conclusion = conclusion

    def __eq__(self, other):
        return (
            isinstance(other, Rule) and
            (self.premise == other.premise) and
            (self.conclusion == other.conclusion)
        )

    def __hash__(self):
        return hash(self.conclusion)

    def __str__(self):
        premise_str = [
            (str(clause)) for clause in sorted(self.premise, key=str)
        ]
        return f"IF {' OR '.join(premise_str)} THEN {self.conclusion}"

    def evaluate_score(self, data):
        """
        Given a list of input neurons and their values, return the combined
        proportion of clauses that satisfy the rule
        """
        total = len(self.premise)
        total_correct_score = 0
        for clause in self.premise:
            if clause.evaluate(data):
                total_correct_score += clause.score

        # Be careful with the always true clause (i.e. empty). In that case, the
        # average score is always 1.
        return total_correct_score/total if total else 1


    @classmethod
    def from_term_set(cls, premise, conclusion, confidence):
        """
        Construct Rule given a single clause as a set of terms and a conclusion
        """
        rule_premise = {
            ConjunctiveClause(terms=premise, confidence=confidence)
        }
        return cls(premise=rule_premise, conclusion=conclusion)

    @classmethod
    def initial_rule(cls, output_layer, output_class, threshold):
        """
        Construct Initial Rule given parameters with default confidence value
        of 1
        """
        rule_premise = ConjunctiveClause(
            terms={Term(
                neuron=Neuron(
                    layer=output_layer,
                    index=output_class,
                ),
                operator='>',
                threshold=threshold,
            )},
            confidence=1
        )
        return cls(premise={rule_premise}, conclusion=output_class)

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        # Every term will be initialized to have a confidence of 1. We will
        # select the minimum across all clauses that use the same term
        term_confidences = defaultdict(lambda: 1)

        for clause in self.premise:
            for term in clause.terms:
                term_confidences[term] = min(
                    term_confidences[term],
                    clause.confidence
                )

        return term_confidences
