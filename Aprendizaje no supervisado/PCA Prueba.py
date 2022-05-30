import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import csv
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

dataset = genfromtxt("dataset9.txt", delimiter=',')
#------------ Arreglo de datos -----------------------


X = dataset[:,0:7]
y = dataset[:,7]

y = y.reshape(960,1)

X_std = StandardScaler().fit_transform(X)



mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('\nEigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)


u, s, v = np.linalg.svd(X_std.T)

print('\nSingular Vector Descomposition \n%s' %u)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('\nEverything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1),
                      eig_pairs[1][1].reshape(7,1)))

print('\nProjection Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

# Sklearn
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
print(X_std)
print('*******************************************')
print(Y_sklearn)




