import random
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # pip install scikit-learn
import os
import pandas as pd
import math
import csv
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
style.use('ggplot')


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


#Desde este punto inicia el Codigo Kmeans




plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c='black', label='Unclustered data')
plt.title('Gráfico de los datos')
plt.show()

m = Y_sklearn.shape[0] # Cantidad de muestras que hay en la base de datos
n = Y_sklearn.shape[1] # Cantidad de catacterísticas en la base de datos
iteraciones = 200

K = 2 # Cantidad de centros o clusters

# Generación de los centroides

centroides = np.array([]).reshape(n, 0)
for i in range(K):
    rand = random.randint(0, m-1)
    centroides = np.c_[centroides, Y_sklearn[rand]]

output = {}

# Cálculo de la distancia euclidiana entre los puntos que constituyen la base de datos

distanciaEuclidiana = np.array([]).reshape(m, 0)

for k in range(K):
    tempDist = np.sqrt(np.sum((Y_sklearn - centroides[:, k])**2, axis=1)) # Aplicarle raiz cuadrada
    distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist]

distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1

Y_kmeans = {}

# Asignamos los puntos a un cluster con base a su distancia mínima

for k in range(K):
    Y_kmeans[k+1] = np.array([]).reshape(2, 0)

for i in range(m):
    Y_kmeans[distanciaMinima[i]] = np.c_[Y_kmeans[distanciaMinima[i]], Y_sklearn[i]]

for k in range(K):
    Y_kmeans[k+1] = Y_kmeans[k+1].T

for k in range(K):
    centroides[:, k] = np.mean(Y_kmeans[k+1], axis=0)

for i in range(iteraciones):
    distanciaEuclidiana = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sqrt(np.sum((Y_sklearn - centroides[:, k])**2, axis=1)) # Aplicarle raiz cuadrada
        distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist]
    distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1
    
    Y_kmeans = {}
    for k in range(K):
        Y_kmeans[k+1] = np.array([]).reshape(2, 0)

    for i in range(m):
        Y_kmeans[distanciaMinima[i]] = np.c_[Y_kmeans[distanciaMinima[i]], Y_sklearn[i]]

    for k in range(K):
        Y_kmeans[k+1] = Y_kmeans[k+1].T

    for k in range(K):
        centroides[:, k] = np.mean(Y_kmeans[k+1], axis=0)
    
    output = Y_kmeans

color = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['cluster1', 'cluster2', 'cluster3','cluster4','cluster5']

for k in range(K):
    plt.scatter(output[k+1][:,0], output[k+1][:,1], c=color[k], label=labels[k])
    plt.show()

plt.scatter(centroides[0, :], centroides[1, :], s=300, c='yellow', label='centroides')
plt.show()

