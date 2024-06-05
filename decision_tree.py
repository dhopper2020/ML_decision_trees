import sys
import re
import math
from node import *

"""
Project done by Dylan Hopper and Shoshanah Bernhardt
"""

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if p <= 0 or p >= 1:
        return 0
    tmp = -p * math.log(p, 2) - (1 - p) * math.log((1 - p), 2)
    return tmp


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    if pxi == 0:
        return 0
    if total-pxi == 0:
        return 0
    if total == 0:
        return 0

    parent_entropy = entropy(py / total)

    qy_qxi = py - py_pxi

    branch_one = entropy(py_pxi / pxi)
    branch_two = entropy(qy_qxi / (total - pxi))

    gain = parent_entropy - (pxi / total) * branch_one - ((total - pxi) / total) * branch_two
    return gain


def partition_data(data, index):
    data_zero = []
    data_one = []

    for x in data:
        if x[index] == 0:
            data_zero.append(x)
        elif x[index] == 1:
            data_one.append(x)

    return data_zero, data_one


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return data, varnames


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

def get_py(data):
    y_count = 0
    for i in data:
        if i[-1] == 1:
            y_count += 1
    return y_count


def get_pxi(data, index):
    pxi = 0

    for i in data:
        if i[index] == 1:
            pxi += 1
    return pxi


def get_py_pxi(data, index):
    py_pxi = 0

    for i in data:
        if i[index] == 1 and i[-1] == 1:
            py_pxi += 1
    return py_pxi


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    total = len(data)
    py = get_py(data)
    gain = ["", 0]

    if len(data) == 0:
        return Leaf(varnames, 0)
    if py == total:
        return Leaf(varnames, 1)
    elif py == 0:
        return Leaf(varnames, 0)

    # go through varnames and get the attribute with the highest infogain, store it in gain with gain[0] = string name
    # and gain[1] = infogain number
    # loop through n-1, use slice
    for i, name in enumerate(varnames[:-1]):
        pxi = get_pxi(data, i)
        py_pxi = get_py_pxi(data, i)
        tmp = infogain(py_pxi, pxi, py, total)
        if tmp > gain[1]:
            gain[0] = i
            gain[1] = tmp

    if gain[1] <= 0:
        if 2 * py > total:
            return Leaf(varnames, 1)
        else:
            return Leaf(varnames, 0)

    data_zero, data_one = partition_data(data, gain[0])

    # recurse return a split node that will call build tree with the two data sets created and either varnames
    return Split(varnames, gain[0], build_tree(data_zero, varnames), build_tree(data_one, varnames))


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)

    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
