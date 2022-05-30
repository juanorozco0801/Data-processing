# coding=utf-8
import random
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # pip install scikit-learn
style.use('ggplot')

X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=10, shuffle=True)

plt.scatter(X[:, 0], X[:, 1], c='black', label='Unclustered data')
plt.title('Gráfico de los datos')
plt.show()

m = X.shape[0] # Cantidad de muestras que hay en la base de datos
n = X.shape[1] # Cantidad de catacterísticas en la base de datos
iteraciones = 100

K = 2 # Cantidad de centros o clusters

# Generación de los centroides

centroides = np.array([]).reshape(n, 0)
for i in range(K):
    rand = random.randint(0, m-1)
    centroides = np.c_[centroides, X[rand]]

output = {}

# Cálculo de la distancia euclidiana entre los puntos que constituyen la base de datos

distanciaEuclidiana = np.array([]).reshape(m, 0)

for k in range(K):
    tempDist = np.sqrt(np.sum((X - centroides[:, k])**2, axis=1)) # Aplicarle raiz cuadrada
    distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist]

distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1

Y = {}

# Asignamos los puntos a un cluster con base a su distancia mínima

for k in range(K):
    Y[k+1] = np.array([]).reshape(2, 0)

for i in range(m):
    Y[distanciaMinima[i]] = np.c_[Y[distanciaMinima[i]], X[i]]

for k in range(K):
    Y[k+1] = Y[k+1].T

for k in range(K):
    centroides[:, k] = np.mean(Y[k+1], axis=0)

for i in range(iteraciones):
    distanciaEuclidiana = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sqrt(np.sum((X - centroides[:, k])**2, axis=1)) # Aplicarle raiz cuadrada
        distanciaEuclidiana = np.c_[distanciaEuclidiana, tempDist]
    distanciaMinima = np.argmin(distanciaEuclidiana, axis=1) + 1
    
    Y = {}
    for k in range(K):
        Y[k+1] = np.array([]).reshape(2, 0)

    for i in range(m):
        Y[distanciaMinima[i]] = np.c_[Y[distanciaMinima[i]], X[i]]

    for k in range(K):
        Y[k+1] = Y[k+1].T

    for k in range(K):
        centroides[:, k] = np.mean(Y[k+1], axis=0)
    
    output = Y

color = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['cluster1', 'cluster2', 'cluster3','cluster4','cluster5']

for k in range(K):
    plt.scatter(output[k+1][:,0], output[k+1][:,1], c=color[k], label=labels[k])
    plt.show()

plt.scatter(centroides[0, :], centroides[1, :], s=300, c='yellow', label='centroides')
plt.show()