"""
For each class
    for each rule for that class
        cc= number of training examples classified correctly using the rule
        todo: should this be n classified correctly / n training examples of that class
        ic = numberin correctly classified
        k=4
        rl = rule length i.e. the number of terms in the rule

"""
import pickle


from ..rules.rule import Rule
from ..rules.term import Neuron

k = 4


def rank_rule(rules, X_train, y_train, use_rl: bool):
    """

    Args:
        rules: The whole ruleset extracted (set of dnf rules for each class)
        X_train: train data
        y_train: test data
        use_rl: if true perform RF+HC-CMPR else RF+HC

    Returns:

    """

    for class_rule in rules:

        # Each run of rule extraction return a DNF rule for each output class
        rule_output = class_rule.get_conclusion()

        # Each clause in the dnf rule is considered a rule for this output class
        for clause in class_rule.get_premise():
            cc = ic = 0
            rl = len(clause.get_terms())

            # Iterate over all items in the training data
            for i in range(0, len(X_train)):
                # Map of Neuron objects to values from input data. This is the form of data a rule expects
                neuron_to_value_map = {Neuron(layer=0, index=j): X_train[i][j]
                                       for j in range(len(X_train[i]))}

                # if rule predicts the correct output class
                if clause.evaluate(data=neuron_to_value_map):
                    if rule_output.encoding == y_train[i]:
                        cc += 1
                    else:
                        ic += 1


            # Compute rule rank_score
            if cc + ic == 0:
                rank_score = 0
            else:
                rank_score = ((cc - ic) / (cc + ic)) + cc / (ic + k)

            if use_rl:
                rank_score += cc / rl

            # print('cc: %d, ic: %d, rl: %d  rankscroe: %f' % (cc, ic, rl, rank_score))

            # Save rank score
            clause.set_rank_score(rank_score)





