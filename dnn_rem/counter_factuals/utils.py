from scipy.spatial import distance
import numpy as np
import random
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dnn_rem.evaluate_rules.metrics import fidelity
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight


def class_based_instances(ruleset, X_train):
    instances_of_the_same_class = {}
    y_predicted = ruleset.predict(X_train)
    for x, y in zip(list(range(0, len(X_train))), y_predicted):
        instances_of_the_same_class.setdefault(y, []).append(x)
    return instances_of_the_same_class


def feature_match_distance(p_1, p_2):
    count = 0
    for p_1_i, p_2_i in zip(p_1, p_2):
        if round(p_1_i, 2) == round(p_2_i, 2):
            count += 1
    return count


def euclidean_distance(p_1, p_2):
    return round(distance.euclidean(p_1, p_2), 3)


def datapoint_to_explanation_map(ruleset, points):
    points_exp_dict = {}
    all_rules = []

    for rules in sorted(ruleset.rules, key=str):
        for rule in sorted(rules.premise, key=str):
            all_rules.append(rule)

    for p, p_id in zip(points, range(len(points))):
        satis_rules = []
        for rule, rule_id in zip(all_rules, range(len(all_rules))):
            if len(rule.terms) > 0 and rule.evaluate(ruleset._get_named_dictionary(p)):
                satis_rules.append(rule_id)
        points_exp_dict[p_id] = satis_rules

    return points_exp_dict


def explanation_match_matrix(X_train, X_test, train_exp_dict, test_exp_dict):
    matrix = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            matrix[i][j] = len(set(test_exp_dict[i]).intersection(set(train_exp_dict[j])))
    return matrix


def normalised_explanation_match_matrix(X_train, X_test, train_exp_dict, test_exp_dict):
    matrix = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            total = len(set(test_exp_dict[i])) + len(set(train_exp_dict[j]))
            common = len(set(test_exp_dict[i]).intersection(set(train_exp_dict[j])))
            matrix[i][j] = common / total
    return matrix


def explanation_match_distance(p_1_i, p_2_i, matrix):
    return matrix[p_1_i][p_2_i]


def conf_interval(x, conf):
    m = np.mean(x)
    if np.all(x == x[0]):
        lo_x = hi_x = m
    else:
        xt, lmbda = stats.boxcox(x)
        lo_xt, hi_xt = stats.t.interval(1 - conf, len(xt) - 1, loc=np.mean(xt), scale=stats.sem(xt))
        lo_x, hi_x = inv_boxcox(lo_xt, lmbda), inv_boxcox(hi_xt, lmbda)
    return m-lo_x, hi_x-m


def evaluate_local_model(X_train,
                         X_test,
                         x_test_index,
                         instances_of_the_same_class,
                         ruleset,
                         matrix,
                         neighbour_finder="random",
                         subsample_percent=0.1,
                         depth_of_tree=5,
                         seed=87):
    np.random.seed(seed)
    random.seed(seed)

    instances_of_the_same_class_new = {}

    for cls in instances_of_the_same_class:
        distances = []
        explanation_match_distances = []
        euclidean_distances = []

        number_of_neighbours = int(len(instances_of_the_same_class[cls]) * subsample_percent)
        X_train_cls = np.array(X_train[instances_of_the_same_class[cls]])

        for x_train_index, x_train in zip(instances_of_the_same_class[cls], X_train_cls):
            if neighbour_finder == "feature_match":
                distances.append(feature_match_distance(X_test[x_test_index], x_train))
            elif neighbour_finder == "explanation_match":
                distances.append(explanation_match_distance(x_test_index, x_train_index, matrix))
            elif neighbour_finder == "euclidean_distance":
                distances.append(euclidean_distance(X_test[x_test_index], x_train))
            elif neighbour_finder == "mix":
                explanation_match_distances.append(explanation_match_distance(x_test_index, x_train_index, matrix))
                euclidean_distances.append(euclidean_distance(X_test[x_test_index], x_train))
            else:
                indices = random.sample(instances_of_the_same_class[cls], number_of_neighbours)

        if neighbour_finder != "random":
            if neighbour_finder == "euclidean_distance":
                sort_result = sorted(zip(distances, X_train_cls, instances_of_the_same_class[cls]), key=lambda x: x[0])
                _, _, indices = zip(*sort_result)
                instances_of_the_same_class_new[cls] = list(indices)[:number_of_neighbours]
            elif neighbour_finder == "mix":
                number_of_neighbours = int(len(instances_of_the_same_class[cls]) * subsample_percent * 0.5)
                explanation_sort_result = sorted(
                    zip(explanation_match_distances, X_train_cls, instances_of_the_same_class[cls]), key=lambda x: x[0],
                    reverse=True)
                _, _, explanation_indices = zip(*explanation_sort_result)

                euclidean_sort_result = sorted(zip(euclidean_distances, X_train_cls, instances_of_the_same_class[cls]),
                                               key=lambda x: x[0])
                _, _, euclidean_indices = zip(*euclidean_sort_result)

                instances_of_the_same_class_new[cls] = list(explanation_indices)[:number_of_neighbours]
                instances_of_the_same_class_new[cls].extend(list(euclidean_indices)[:number_of_neighbours])

                if len(instances_of_the_same_class_new[cls]) > len(set(instances_of_the_same_class_new[cls])):
                    diff = len(instances_of_the_same_class_new[cls]) - len(set(instances_of_the_same_class_new[cls]))

                    # We use the euclidean as opposed to the explanation to make up for the numbers
                    remaining_euclidean = list(euclidean_indices)[number_of_neighbours:]
                    non_repetetive_remaning = [entry for entry in remaining_euclidean if
                                               entry not in instances_of_the_same_class_new[cls]]
                    instances_of_the_same_class_new[cls].extend(non_repetetive_remaning[:diff])
            else:
                sort_result = sorted(zip(distances, X_train_cls, instances_of_the_same_class[cls]), key=lambda x: x[0],
                                     reverse=True)
                _, _, indices = zip(*sort_result)
                instances_of_the_same_class_new[cls] = list(indices)[:number_of_neighbours]
        else:
            instances_of_the_same_class_new[cls] = list(indices)[:number_of_neighbours]

    neighbours = [item for sublist in list(instances_of_the_same_class_new.values()) for item in sublist]

    test_sample_pred = ruleset.predict(X_test[x_test_index])
    X_neighbours = X_train[list(set(neighbours))]
    y_neighbours = ruleset.predict(X_neighbours)

    X_neighbours_train, X_neighbours_test, y_neighbours_train, y_neighbours_test = train_test_split(X_neighbours,
                                                                                                    y_neighbours,
                                                                                                    test_size=0.2,
                                                                                                    random_state=seed)
    weight = dict(
        zip(np.unique(y_neighbours_train), class_weight.compute_class_weight('balanced', np.unique(y_neighbours_train),
                                                                             y_neighbours_train)))
    clf = DecisionTreeClassifier(max_depth=depth_of_tree, class_weight=weight, random_state=seed)
    clf = clf.fit(X_neighbours_train, y_neighbours_train)
    x_test_local = clf.predict([X_test[x_test_index]])
    hit_score = fidelity(x_test_local, test_sample_pred)
    y_neighbour_test_local = clf.predict(X_neighbours_test)
    fidelity_score = fidelity(y_neighbour_test_local, y_neighbours_test)

    if len(np.unique(y_neighbours_test)) <= 2 and len(np.unique(y_neighbour_test_local)) <= 2:
        auc_score = roc_auc_score(y_neighbours_test, y_neighbour_test_local)
    else:
        max_unique = max(np.max(y_neighbours_test), np.max(y_neighbour_test_local))
        auc_score = roc_auc_score(
            to_categorical(
                y_neighbours_test,
                num_classes=max_unique + 1
            ),
            to_categorical(
                y_neighbour_test_local,
                num_classes=max_unique + 1
            ),
            multi_class="ovr",
            average='micro',

        )

    return hit_score, fidelity_score


def result_summary(results, number_of_methods, number_of_seeds, num_of_test_samples):
    results_reshaped = np.array(results).reshape(number_of_methods, number_of_seeds, num_of_test_samples, 2)
    methods_hit = []
    methods_fid = []

    for i in range(0, number_of_methods):
        seeds_hit = []
        seeds_fid = []
        for j in range(0, number_of_seeds):
            hit = np.sum(results_reshaped[i][j][:, 0]) / num_of_test_samples
            seeds_hit.append(hit)
            fid = np.mean(results_reshaped[i][j][:, 1])
            seeds_fid.append(fid)

        mean_hit = np.mean(seeds_hit)
        lo_hit, hi_hit = conf_interval(np.array(seeds_hit), 0.05)
        methods_hit.append([round(mean_hit, 4), round(lo_hit, 4), round(hi_hit, 4)])

        mean_fid = np.mean(seeds_fid)
        lo_fid, hi_fid = conf_interval(np.array(seeds_fid), 0.05)
        methods_fid.append([round(mean_fid, 4), round(lo_fid, 4), round(hi_fid, 4)])

    return methods_hit, methods_fid
