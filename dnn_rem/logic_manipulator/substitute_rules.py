"""
Methods for making rule substitution.
"""

import itertools

from ..rules.clause import ConjunctiveClause
from ..rules.rule import Rule


def substitute(total_rule, intermediate_rules):
    """
    Substitute the intermediate rules from the previous layer into the total
    rule.

    :param Rule total_rule: The receiver rule of the beta reduction we are about
                            to make.
    :param Ruleset intermediate_rules: The set of intermediate rules which we
                                       want to substitute in the given total
                                       rule.

    :returns Rule: a new rule equivalent to making the substitution of the given
                   intermediate rules into total_rule.
    """
    new_premise_clauses = set()

    # for each clause in the total rule
    for old_premise_clause in total_rule.get_premise():
        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = []
        for old_premise_term in old_premise_clause.get_terms():
            clauses_to_append = \
                intermediate_rules.get_rule_premises_by_conclusion(
                    old_premise_term
                )
            if clauses_to_append:
                conj_new_premise_clauses.append(clauses_to_append)

        # When combined into a cartesian product, get all possible conjunctive
        # clauses for merged rule
        # Itertools implementation does not build up intermediate results in
        # memory
        conj_new_premise_clauses_combinations = itertools.product(
            *tuple(conj_new_premise_clauses)
        )

        # given tuples of ConjunctiveClauses that are all now conjunctions,
        # union terms into a single clause
        for premise_clause_tuple in conj_new_premise_clauses_combinations:
            new_clause = ConjunctiveClause()
            for premise_clause in premise_clause_tuple:
                new_clause = new_clause.union(premise_clause)
            new_premise_clauses.add(new_clause)

    return Rule(
        premise=new_premise_clauses,
        conclusion=total_rule.get_conclusion(),
    )
