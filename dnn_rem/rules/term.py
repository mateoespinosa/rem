"""
Represent components that make up a rule. All immutable and hashable.
"""

from enum import Enum
import numpy as np


class TermOperator(Enum):
    GreaterThan = '>'
    LessThanEq = '<='
    # Equal = '='

    def __str__(self) -> str:
        return self.value

    def negate(self):
        # Negate term
        if self is self.GreaterThan:
            return self.LessThanEq
        if self is self.LessThanEq:
            return self.GreaterThan

    def eval(self):
        # Return evaluation operation for term operator
        if self is self.GreaterThan:
            return lambda x, y: x > y
        if self is self.LessThanEq:
            return lambda x, y: np.logical_or(np.isclose(x, y), (x < y))

    def most_general_value(self, values):
        # Given a list of values, return the most general depending on the
        # operator
        if self is self.GreaterThan:
            return max(values)
        if self is self.LessThanEq:
            return min(values)


class Term(object):
    """
    Represent a condition indicating if activation value of variable is
    above/below a threshold.

    Immutable and Hashable.
    """
    def __init__(self, variable, operator, threshold):
        self.variable = variable
        self.threshold = threshold
        self.operator = TermOperator(operator)

    def __str__(self):
        return f'({self.variable} {self.operator} {self.threshold})'

    def __eq__(self, other):
        return (
            isinstance(other, Term) and
            (self.variable == other.variable) and
            (self.operator == other.operator) and
            (np.isclose(self.threshold, other.threshold))
        )

    def __hash__(self):
        return hash((self.variable, self.operator, self.threshold))

    def negate(self):
        """
        Return term with opposite sign
        """
        return Term(
            self.variable,
            str(self.operator.negate()),
            self.threshold
        )

    def apply(self, value):
        """
        Apply condition to a value
        """
        return self.operator.eval()(value, self.threshold)

    def get_neuron_index(self):
        """
        Return index of variable specified in the term
        """
        return self.variable.index

    def to_json(self):
        result = {}
        result["variable"] = self.variable
        result["threshold"] = self.threshold
        result["operator"] = str(self.operator)
        return result
