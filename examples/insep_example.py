#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'..')
import lin_sep
from lin_sep import linearly_separable

plt.style.use('ggplot')

# read linearly separable data
data = np.loadtxt('insep.txt')

X = data[:, 0:-1]
y = data[:, -1]

success, result = linearly_separable(X, y)

# plot
print "Data is linearly separable: {}".format(success)

c1_inds = np.where(y == 0)[0]
c2_inds = np.where(y == 1)[0]

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.savefig("insep_plot.png", dpi=300)
