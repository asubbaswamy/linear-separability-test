#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from lin_sep import linearly_separable

plt.style.use('ggplot')

# read linearly separable data
data = np.loadtxt('separable.txt')

X = data[:, 0:-1]
y = data[:, -1]

success, result = linearly_separable(X, y)

h1, h2, beta = result['x']

in_ = np.linspace(0, 1, 1000)
out = beta/h2 + -h1/h2 * in_

# plot
print "Data is linearly separable: {}".format(success)

c1_inds = np.where(y == 0)[0]
c2_inds = np.where(y == 1)[0]

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(in_, out)

plt.savefig("sep_plot.png", dpi=300)
