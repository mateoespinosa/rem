"""
Represent a rule with a premise in Disjunctive Normal Form (DNF) and conclusion
of another term or class conclusion
"""

from .clause import ConjunctiveClause
from .term import Term, Neuron
from . import DELETE_UNSATISFIABLE_CLAUSES_FLAG
from dnn_rem.logic_manipulator.satisfiability import \
    remove_unsatisfiable_clauses


class OutputClass(object):
    """
    Represents the conclusion of a given rule. Immutable and Hashable.

    Each output class has a name and its relevant encoding in the network
    i.e. which output neuron it corresponds to
    """
    __slots__ = ['name', 'encoding']

    def __init__(self, name: str, encoding: int):
        super(OutputClass, self).__setattr__('name', name)
        super(OutputClass, self).__setattr__('encoding', encoding)

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
    __slots__ = ['premise', 'conclusion']

    def __init__(self, premise, conclusion):
        if DELETE_UNSATISFIABLE_CLAUSES_FLAG:
            premise = remove_unsatisfiable_clauses(clauses=premise)

        # if DELETE_REDUNDANT_CLAUSES_FLAG:
        #     premise = self.delete_redundant_clauses(clauses=premise)

        super(Rule, self).__setattr__('premise', premise)
        super(Rule, self).__setattr__('conclusion', conclusion)

    def get_premise(self):
        return self.premise

    def get_conclusion(self):
        return self.conclusion

    def __eq__(self, other):
        return (
            isinstance(other, Rule) and
            (self.premise == other.premise) and
            (self.conclusion == other.conclusion)
        )

    def __hash__(self):
        return hash((self.conclusion))

    def __str__(self):
        premise_str = [(str(clause)) for clause in self.premise]
        return f"IF {' OR '.join(premise_str)} THEN {self.conclusion}"

    def evaluate_rule_by_majority_voting(self, data):
        """
        Given a list of input neurons and their values, return the combined
        proportion of clauses that satisfy the rule
        """
        total = len(self.premise)
        n_satisfied_clauses = 0
        for clause in self.premise:
            if clause.evaluate(data):
                n_satisfied_clauses += 1

        # Be careful with the always true clause (i.e. empty)
        return n_satisfied_clauses/total if total else 1

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
        rule_conclusion = output_class

        return cls(premise={rule_premise}, conclusion=rule_conclusion)

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        term_confidences = {}

        for clause in self.premise:
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
