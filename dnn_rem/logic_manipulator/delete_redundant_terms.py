"""
Methods for redundancy removal in sets of clauses.
"""

from ..rules.term import Term, TermOperator
from .utils import terms_set_to_variable_dict


def remove_redundant_terms(terms):
    """
    Remove redundant terms from a clause, returning only the necessary terms
    """
    # We generate a map {variable name (str): {TermOperator: [Float]}}
    variable_conditions = terms_set_to_variable_dict(terms)
    necessary_terms = set()

    # Find most general variable thresholds, range as general as possible,
    # for '>' keep min, for '<=' keep max
    for variable, threshold_map in variable_conditions.items():
        for TermOp in TermOperator:
            if threshold_map[TermOp]:  # if non-empty list
                necessary_terms.add(
                    Term(
                        variable=variable,
                        operator=TermOp,
                        threshold=TermOp.most_general_value(
                            threshold_map[TermOp]
                        )
                    )
                )

    return necessary_terms
