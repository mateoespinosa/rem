import pickle
import copy
import numpy as np
from ..rules.rule import Rule



def eliminate_rules(rules, n):
    """
    :param Iterable[Rule] rules:    The rules that will be subjected to elimination
    :param int n:   The percentage of rules that will be eliminated. n = 0.7 eliminates 70% of the rules
    Returns Iterable[Rule] rule:    The remaining rules after elimination

    """

    updated_class_Rule_list = []
    for class_rule in rules:
        clause_list = []
        clause_score = []
        for clause in class_rule.get_premise():
            clause_list.append(clause)
            clause_score.append(clause.get_rank_score())

        numToEliminate = round(len(clause_list) * n)
        remained_clause = copy.deepcopy(clause_list)
        for i in range(0, numToEliminate):
            index = np.argmin(clause_score)
            del remained_clause[index]
            del clause_score[index]

        updated_class_Rule = Rule(premise=(set(remained_clause)), conclusion=class_rule.get_conclusion())
        updated_class_Rule_list.append(updated_class_Rule)

    rules.clear()
    rules.update(updated_class_Rule_list)

    return rules





