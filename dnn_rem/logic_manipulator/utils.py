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

    :returns Dict[Neuron, Dict[TermOperator, List[float]]]: the corresponding
                                                            dictionary of neuron
                                                            values.
    """
    # Convert set of conditions into dictionary
    neuron_conditions = {}

    for term in terms:
        if not term.get_neuron() in neuron_conditions:  # unseen Neuron
            neuron_conditions[term.get_neuron()] = {
                TermOp: []
                for TermOp in TermOperator
            }
        neuron_conditions[(term.get_neuron())][term.get_operator()].append(
            term.get_threshold()
        )

    # Return {Neuron: {TermOperator: [Float]}}
    return neuron_conditions
