#!/usr/bin/python

import sys
import getopt
import numpy as np
from scipy.optimize import linprog

"""Technique explained in blog found here: 
    https://www.joyofdata.de/blog/testing-linear-separability-linear-programming-r-glpk/
"""

def linearly_separable(X, y):
    """Determines if a y is linearly separable with respect to X

    Args:
        X: np matrix containing the features of the observations
        y: np array with the (assumed binary) labels of the observations.

    Returns:
        True, if y is linearly separable with respect to X.
        False, otherwise.
    """

    # how many variables are there
    num_vars = X.shape[1]

    # nothing to maximize; just checking feasibility through constraints
    objective = np.zeros(num_vars + 1)

    # split X into two classes of y
    y_values = np.unique(y).flatten()

    y0_indices = np.where(y == y_values[0])[0]
    y1_indices = np.where(y == y_values[1])[0]

    X1 = X[y0_indices, :]
    X2 = X[y1_indices, :]

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n = n1 + n2
    
    rhs = np.ones(n) * -1.
    # class 1
    lhs1 = np.hstack((-1.* X1, np.ones(n1).reshape(-1, 1)))
    # class 2
    lhs2 = np.hstack((X2, -1. * np.ones(n2).reshape(-1, 1)))
    
    lhs = np.vstack((lhs1, lhs2))

    result = linprog(objective, A_ub=lhs, b_ub=rhs, bounds=(None, None))

    return result['success'], result

def main(argv):
    input_file = ''
    label_column = -1
    try:
        opts, args = getopt.getopt(argv, "hi:l", ["help", "input_file=", "label_column"])
    except getopt.GetoptError:
        print "lin-sep.py -i <inputfile> [-l <label_column>]\n -l label_column"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print """lin-sep.py -i <inputfile> [-l <label_column>]\n -l label_column
            \n Label_column assumes 0-based indexing. If unspecified, assumed to be the last column
            \n -i input_file
            \n Input file, csv with no headers. Should be readable by 'numpy.loadtxt'"""
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-l", "--label_column"):
            label_column = arg

    print "Input file name: {}".format(input_file)
    print "Label column is {}".format(label_column)

    data = np.loadtxt(input_file)

    y = data[:, label_column]

    X = np.delete(data, label_column, axis=1)

    print linearly-separable(X, y)[0]



if __name__ == '__main__':
    main(sys.argv[1:])
