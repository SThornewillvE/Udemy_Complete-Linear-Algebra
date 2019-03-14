# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

plt.style.use("seaborn")

X, y = make_regression(n_samples=100, n_features=1, noise = 50)


plt.scatter(X, y)

A = np.zeros(shape=(100, 2))
A[:, 0] = X[:, 0]
A[:, 1] = y

U, D, V_T = np.linalg.svd(A)

A_pred = U[:, 0].reshape((100, 1)).dot(D[0]).dot(V_T[0, :].reshape((1, 2)))
X_pred = A_pred[:, 0]
y_pred = A_pred[:, 1]

plt.scatter(X, y, label = 'generated data')
plt.scatter(X_pred, y_pred, label = 'first PC')
plt.title('Projecting 2d data onto line')
plt.legend()
plt.show()