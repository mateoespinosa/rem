"""
Represent components that make up a rule. All immutable and hashable.
"""

from enum import Enum
from typing import List


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
        import operator
        if self is self.GreaterThan:
            return operator.gt
        if self is self.LessThanEq:
            return operator.le
        # if self is self.Equal:
        #     return operator.eq

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

    __slots__ = ['layer', 'index']

    def __init__(self, layer: int, index: int):
        super(Neuron, self).__setattr__('layer', layer)
        super(Neuron, self).__setattr__('index', index)

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

    def get_index(self):
        return self.index


class Term(object):
    """
    Represent a condition indicating if activation value of neuron is
    above/below a threshold.

    Immutable and Hashable.
    """

    __slots__ = ['neuron', 'operator', 'threshold']

    def __init__(self, neuron, operator, threshold):
        super(Term, self).__setattr__('neuron', neuron)
        super(Term, self).__setattr__('threshold', threshold)

        operator = TermOperator(operator)
        super(Term, self).__setattr__('operator', operator)

    def __str__(self):
        return f'({self.neuron} {self.operator} {self.threshold})'

    def __eq__(self, other):
        return (
            isinstance(other, Term) and
            (self.neuron == other.neuron) and
            (self.operator == other.operator) and
            (self.threshold == other.threshold)
        )

    def __hash__(self):
        return hash((self.neuron, self.operator, self.threshold))

    def negate(self):
        """
        Return term with opposite sign
        """
        return Term(
            self.neuron,
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
        return self.neuron.get_index()

    def get_neuron(self):
        return Neuron(self.neuron.layer, self.neuron.index)

    def get_operator(self):
        return self.operator

    def get_threshold(self):
        return self.threshold


