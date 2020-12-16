"""
Represent components that make up a rule. All immutable and hashable.
"""

from enum import Enum
from typing import List
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


class Neuron(object):
    """
    Represent specific neuron in the neural network. Immutable and Hashable.
    """

    def __init__(self, layer: int, index: int):
        self.layer = layer
        self.index = index

    def __str__(self):
        return f'h_{self.layer},{self.index}'

    def __eq__(self, other):
        return (
            isinstance(other, Neuron) and
            (self.index == other.index) and
            (self.layer == other.layer)
        )

    def __hash__(self):
        return hash((self.layer, self.index))


class Term(object):
    """
    Represent a condition indicating if activation value of neuron is
    above/below a threshold.

    Immutable and Hashable.
    """
    def __init__(self, neuron, operator, threshold):
        self._neuron = neuron
        self.threshold = threshold
        self.operator = TermOperator(operator)

    def __str__(self):
        return f'({self._neuron} {self.operator} {self.threshold})'

    def __eq__(self, other):
        return (
            isinstance(other, Term) and
            (self._neuron == other.neuron) and
            (self.operator == other.operator) and
            (np.isclose(self.threshold, other.threshold))
        )

    def __hash__(self):
        return hash((self._neuron, self.operator, self.threshold))

    def negate(self):
        """
        Return term with opposite sign
        """
        return Term(
            self._neuron,
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
        Return index of neuron specified in the term
        """
        return self._neuron.index

    @property
    def neuron(self):
        # Return a copy of our current neuron
        return Neuron(self._neuron.layer, self._neuron.index)

    @neuron.setter
    def neuron(self, value):
        # Copy the input neuron
        self._neuron = Neuron(value.layer, value.index)
