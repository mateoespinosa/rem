"""
Merge multiple rules of into Disjunctive Normal Form rules

e.g.
if x>1 AND y<3 AND z<1 THEN 1
if x>4 THEN 2
if y<0.4 THEN 2
->
if (x>1 AND y<3 AND z<1) THEN 1
if (x>4) OR (y<0.4) THEN 2
"""
from ..rules.rule import Rule


def merge(rules):
    """
    Given a disjunctive set of rules (rules must be made up of only conjunctive
    terms), this method will return an equivalent rule set in DNF.

    :param Set[Rule] rules: A set of rules we are merging.

    :returns Set[Rule]: An equivalent set of rules to the provided one in DNF.
    """

    # Build Dictionary mapping rule conclusions to premises(= a set of
    # ConjunctiveClauses)
    rule_conclusion_to_premises_map = {}
    for rule in rules:
        premise = rule.premise
        conclusion = rule.conclusion

        assert len(premise) == 1, \
            'Error: all C5 rules must return 1 conjunctive clause'

        if rule.conclusion in rule_conclusion_to_premises_map:
            # Seen conclusion - add rule premise to set of premises for that
            #                   conclusion
            rule_conclusion_to_premises_map[conclusion] = \
                rule_conclusion_to_premises_map[conclusion].union(premise)
        else:
            # Unseen conclusion - initialise dictionary entry with a set of 1
            # conjunctive clauses
            rule_conclusion_to_premises_map[conclusion] = premise

    # Convert this dictionary into a set of rules where each conclusion occurs
    # only once, i.e. all rules are in DNF
    DNF_rules = set()
    for conclusion, premise in rule_conclusion_to_premises_map.items():
        DNF_rules.add(Rule(
            premise=premise,
            conclusion=conclusion
        ))

    return DNF_rules
