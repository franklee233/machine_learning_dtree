# EECS 349 2014 Fall PS1
# Hangbin Li, HLL932

import sys
import csv
import random
import math
import copy


def doTrial():
# multiple trials, i.e., multiple invocations
    print 'TRIAL NUMBER:',  trial_num, '\n--------------------\n\n', 'DECISION TREE STRUCTURE:'

    # TASK: 2) Divide the set of examples into a training set and a testing
    # set by randomly selecting the number of examples for the training set
    # specified in the command-line input <trainingSetSize>. Use the remainder
    # for the testing set.
    train_set = []
    test_set = copy.deepcopy(instances)
    for i in range(0, train_set_size):
        random_index = int(math.floor(random.random() * len(test_set)))
        train_set.append(test_set[random_index])
        del test_set[random_index]

    # when debugging, this disables random to check specific result
    # train_set = copy.deepcopy(instances[0:train_set_size])
    # test_set = copy.deepcopy(instances[train_set_size:])
    # print len(train_set), len(test_set)

    # calculate prior probability
    prior_prob_true_count = 0
    for i in train_set:
        if i['CLASS']:
            prior_prob_true_count += 1
    prior_prob_true = float(prior_prob_true_count) / len(train_set)
    # print prior_prob_true

    # TASK: 3) Estimate the expected prior probability of TRUE and FALSE
    # classifications, based on the examples in the training set.
    prior_pos_num = 0.0
    for i in train_set:
        prior_pos_num += (i['CLASS'])
    prior_prob_true = prior_pos_num / len(train_set)

    selected_attrib = selectSplitAttri(train_set, attributes)
    parent = 'root'
    attributes.append(parent)
    # print len(train_set), attributes, parent, selected_attrib

    # TASK: 4) Construct a decision tree, based on the training set, using the
    # approach described in Chapter 3 of Mitchell's Machine Learning.
    deci_tree = constructTree(
        train_set, attributes, parent, selected_attrib)
    attributes.remove('root')

    # TASK: 5) Classify the examples in the testing set using the decision
    # tree built in step 4.
    classified_test_set = copy.deepcopy(test_set)
    for i in classified_test_set:
        i['CLASS'] = classifyTestSet(deci_tree, i)

    # TASK: 6) Classify the examples in the testing set using just the prior
    # probabilities from step 3.
    prior_test_set = copy.deepcopy(test_set)
    for i in prior_test_set:
        i['CLASS'] = int(random.random() < prior_prob_true)

    # TASK: 7) Determine the proportion of correct classifications made in
    # steps 5 and 6 by comparing the classifications to the correct answers.
    classified_accuracy = calcClassifyAccuracy(
        classified_test_set, test_set)
    global total_classified_accuracy
    total_classified_accuracy += classified_accuracy

    # TASK: 9) Print the results for each trial to an output file to the
    # command line (standard output). The format of the output is specified in
    # the following section.
    print '\n\t\tPercent of test cases correctly classified by a decision tree built with ID3 =',
    print int(classified_accuracy * 100), '%'

    prior_accuracy = calcClassifyAccuracy(prior_test_set, test_set)
    global total_prior_accuracy
    total_prior_accuracy += prior_accuracy
    print '\t\tPercent of test cases correctly classified by using prior probabilities from the training set =',
    print int(prior_accuracy * 100), '% correct classification\n'

    global verbose
    if verbose:
        print '-------------------------training set------------------------'
        for i in train_set:
            print i
        print '\n-------------------------testing set-------------------------'
        for i in test_set:
            print i
        print '\n--------------------classified testing set-------------------'
        for i in classified_test_set:
            print i
        print '\n----------applying prior probability on testing set----------'
        for i in prior_test_set:
            print i
        print


def calcClassifyAccuracy(classified_test_set, reference_set):
# calculate classified accuracy
    # print len(reference_set), len(classified_test_set)
    identical_num = 0
    for i in range(0, len(reference_set)):
        # print classified_test_set[i]['CLASS'], reference_set[i]['CLASS']
        identical_num += (classified_test_set[i]
                          ['CLASS'] == reference_set[i]['CLASS'])
    return float(identical_num) / len(reference_set)


def classifyTestSet(deci_tree, testing_instance):
# using decision tree to classify testing set
    child_value = deci_tree[deci_tree.keys()[0]]
    for i in deci_tree[deci_tree.keys()[0]]:
        if testing_instance[deci_tree.keys()[0]] == i:
            if isinstance(child_value[i], bool):
                bool_class = child_value[i]
                # print bool_class
            else:
                bool_class = classifyTestSet(
                    child_value[i], testing_instance)
    return bool_class


def constructTree(train_set, avail_attribs, parent, selected_attrib):
# idea: replace bool with subtree of attribute:bool
    deci_tree = {}
    deci_tree[selected_attrib] = {}
    # print avail_attribs, parent
    avail_attribs.remove(parent)

    # count how class distributes with +/- +/-
    # pc = positive class, nc = negative class
    train_set_pc = selectDataByAttribBool(
        train_set, 'CLASS', True)
    train_set_nc = selectDataByAttribBool(
        train_set, 'CLASS', False)
    # print len(train_set_pc), len(train_set_nc)
    if not train_set_pc:
        print 'parent:', parent, '-'
        avail_attribs.append(parent)
        return False
    elif not train_set_nc:
        print 'parent:', parent, '-'
        avail_attribs.append(parent)
        return True

    # pa = positive attribute, na = negative attribute
    train_set_pa = selectDataByAttribBool(
        train_set, selected_attrib, True)
    train_set_na = selectDataByAttribBool(
        train_set, selected_attrib, False)

    # print len(train_set_pa), len(train_set_na)
    sub_attribute = {}
    avail_attribs.remove(selected_attrib)
    sub_attribute[True] = selectSplitAttri(
        train_set_pa, avail_attribs)
    sub_attribute[False] = selectSplitAttri(
        train_set_na, avail_attribs)
    print 'parent:', parent, '\tattribute:', selected_attrib, '\ttrueChild:',
    print sub_attribute[True], '\tfalseChild:', sub_attribute[False]

    # recursively left/right construct the tree
    avail_attribs.append(selected_attrib)
    deci_tree[selected_attrib][True] = constructTree(
        train_set_pa, avail_attribs, selected_attrib, sub_attribute[True])
    deci_tree[selected_attrib][False] = constructTree(
        train_set_na, avail_attribs, selected_attrib, sub_attribute[False])
    avail_attribs.append(parent)

    return deci_tree


def selectDataByAttribBool(train_set, selected_attrib, class_bool):
# select partial data from training set depending on the attribute/class bool
    ret_data = []
    for i in train_set:
        if i[selected_attrib] == class_bool:
            ret_data.append(i)
    return ret_data


def selectSplitAttri(train_set, avail_attribs):
# find the minimum entropy attribute
    if not train_set or not avail_attribs:
        return 'leaf'
    attributes_entropy = {}
    selected_attrib = avail_attribs[0]
    selected_entropy = calcEntropy(train_set, avail_attribs[0])
    for attribute in avail_attribs:
        current_entropy = calcEntropy(train_set, attribute)
        # print attribute, current_entropy, len(train_set)
        attributes_entropy[attribute] = current_entropy
        if current_entropy < selected_entropy:
            selected_attrib = attribute
            selected_entropy = current_entropy
    # print attributes_entropy
    # print selected_attrib, selected_entropy
    return selected_attrib


def calcEntropy(train_set, attribute):
# calculate entropy of specific training set & attribute
    # papc = positive attribute positive class,
    # nanc = negative attribute negative class
    papc = panc = napc = nanc = 0
    for i in train_set:
        if i['CLASS'] and i[attribute]:
            papc += 1
        elif i['CLASS'] and not i[attribute]:
            napc += 1
        elif not i['CLASS'] and i[attribute]:
            panc += 1
        elif not i['CLASS'] and not i[attribute]:
            nanc += 1
    # deal with log(0)
    if papc + panc:
        papc_percent = float(papc) / (papc + panc)
    else:
        papc_percent = 0.0
    if papc_percent == 0.0 or papc_percent == 1.0:
        entropy_pa = 0.0
    else:
        entropy_pa = (papc_percent * math.log(papc_percent) +
                      (1 - papc_percent) * math.log(1 - papc_percent))
    if napc + nanc:
        napc_percent = float(napc) / (napc + nanc)
    else:
        napc_percent = 0.0
    if napc_percent == 0.0 or napc_percent == 1.0:
        entropy_na = 0.0
    else:
        entropy_na = (napc_percent * math.log(napc_percent) +
                      (1 - napc_percent) * math.log(1 - napc_percent))

    if papc + panc + napc + nanc:
        pa_percent = float(papc + panc) / (papc + panc + napc + nanc)
        entropy = -(
            pa_percent * entropy_pa + (1 - pa_percent) * entropy_na) / math.log(2)
        return entropy
    else:
        return 0.0


# main
if __name__ == "__main__":
    # process argv
    if len(sys.argv) == 1:
        # input_file_name = 'IvyLeague.txt'
        input_file_name = 'MajorityRule.txt'
        train_set_size = 42
        number_of_trials = 3
        verbose = 0
    else:
        input_file_name = sys.argv[1]
        train_set_size = int(sys.argv[2])
        number_of_trials = int(sys.argv[3])
        verbose = int(sys.argv[4])

    # TASK: 1) Read in the specified text file containing the examples.
    file_handler = open(input_file_name, "rb")
    reader = csv.reader(file_handler, delimiter='\t')
    attributes = reader.next()
    for i in range(0, len(attributes)):
        attributes[i] = attributes[i].strip()

    # instances and attributes are GLOBAL
    instances = []
    for row in reader:
        row_dict = {}
        for i in range(0, len(row)):
            if row[i].strip().lower() == 'true':
                row_dict[attributes[i]] = True
            else:
                row_dict[attributes[i]] = False
        # print row_dict
        instances.append(row_dict)

    del attributes[-1]  # last elm is class

    # total_classified_accuracy and total_prior_accuracy are GLOBAL
    total_prior_accuracy = total_classified_accuracy = 0.0

    # TASK: 8) Steps 2 through 7 constitute a trial. Repeat steps 2 through 7
    # until the number of trials is equal to the value specified in the
    # command-line input <number of trials>.
    for trial_num in range(0, number_of_trials):
        doTrial()

    print 'example file used =', input_file_name
    print 'number of trials =', number_of_trials
    print 'training set size for each trial =', train_set_size
    print 'testing set size for each trial =', len(instances) - train_set_size
    print 'mean performance of decision tree over all trials =',
    print int(total_classified_accuracy / number_of_trials * 100), '% correct classification'
    print 'mean performance of using prior probability derived from the training set =',
    print int(total_prior_accuracy / number_of_trials * 100), '% correct classification'
