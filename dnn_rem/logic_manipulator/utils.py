"""
Helper functions for our logic manipulators.
"""
from ..rules.term import TermOperator


def terms_set_to_neuron_dict(terms):
    """
    Converts a set of terms into a dictionary mapping each neuron in each term
    to a map between operators ('>' and '<=') and thresholds used in those
    neurons with those operators.

    :param Iterable[Terms] terms: The terms which will source our dictionary.

    :returns Dict[Neuron, Dict[TermOperator, Set[float]]]: the corresponding
        dictionary of neuron values.
    """
    # Convert set of conditions into dictionary
    neuron_conditions = {}

    for term in terms:
        neuron = term.neuron
        if not neuron in neuron_conditions:  # unseen Neuron
            neuron_conditions[neuron] = {
                TermOp: set()
                for TermOp in TermOperator
            }
        neuron_conditions[neuron][term.operator].add(term.threshold)

    # Returns {Neuron: {TermOperator: Set[Float]}}
    return neuron_conditions
