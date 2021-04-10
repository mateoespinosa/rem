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


def global_most_general_replacement(rule):
    """
    Removes redundant terms in a "global" fashion (i.e., across clauses rather
    than on a single clause basis). This means that by the end, all clauses in
    the rule can split on a feature at most twice (i.e., one lower and one upper
    bound)

    :param Rule rule:  The rule which we will do the replacement in.
    """
    global_terms = set()
    for clause in rule.premise:
        global_terms |= clause.terms

    variable_conditions = terms_set_to_variable_dict(global_terms)
    for clause in rule.premise:
        for term in clause.terms:
            op = term.operator
            new_threshold = op.most_general_value(
                variable_conditions[term.variable][op]
            )
            if term.threshold != new_threshold:
                term.threshold = new_threshold
