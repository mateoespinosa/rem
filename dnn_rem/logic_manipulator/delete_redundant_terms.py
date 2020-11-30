"""
Methods for redundancy removal in sets of clauses.
"""

from ..rules.term import Term, TermOperator
from .utils import terms_set_to_neuron_dict


def remove_redundant_terms(terms):
    """
    Remove redundant terms from a clause, returning only the necessary terms
    """
    # We generate a map {Neuron: {TermOperator: [Float]}}
    neuron_conditions = terms_set_to_neuron_dict(terms)
    necessary_terms = set()

    # Find most general neuron thresholds, range as general as possible,
    # for '>' keep min, for '<=' keep max
    for neuron, threshold_map in neuron_conditions.items():
        for TermOp in TermOperator:
            if threshold_map[TermOp]:  # if non-empty list
                necessary_terms.add(
                    Term(
                        neuron=neuron,
                        operator=TermOp,
                        threshold=TermOp.most_general_value(
                            threshold_map[TermOp]
                        )
                    )
                )

    return necessary_terms
