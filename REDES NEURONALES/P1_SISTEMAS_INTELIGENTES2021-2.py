# ---------------- SISTEMAS INTELIGENTES -------------------
#                    Perceptron multicapa 
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import csv


dataset = np.genfromtxt("dataset9.txt", delimiter=",")
print(dataset)
np.random.shuffle(dataset)
print(dataset)




#Arreglo de datos 
X = (np.array([dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3], dataset[:, 4], dataset[:, 5], dataset[:, 6]])) #Arreglo de datos de entrada
Y = np.array([dataset[:, 7]])   #Arreglo de datos de salida
nX = X.shape[0] #Numero de caracteristicas
nY = Y.shape[0] #Numero de salidas




#Ingreso de datos de usuario
nL = int(input('Ingrese numero de capas ocultas: '))
nH = int(input('Ingrese numero de neuronas por capa Oculta: '))
epoch = int(input('Ingrese numero de Iteraciones: '))
learningRate = float(input('Ingrese tasa de aprendizaje: '))





#Inicializacion de parametros para la red neuronal
W = {}#diccionario de los pesos
b = {}#diccionario bias
A = {}#activaciones
Z = {}#sumatoria
dZ = {}#derivada de sumatoria
dW = {}#derivada de los pesos
db = {}#deribada del bias
dA = {}#derivada de las activaciones
grads = {}#gradientes




#Función de activación sigmoide 1/(1+(e^-z))
def sigmoid(z):
   return 1 / (1 + np.exp(-z))

# Funcion de activacion Tangente hiperbolica
def Hiperbolica(z):
    return (((2)/(1+np.exp(-2*z)))-1)




#Inicializacion de parametros
def initializeParameters(nX, nH, nY, nL):
    # Creación de la estructura de la red neuronal
    for i in range(1, nL + 1):
        if i == 1:#Primera capa
            W[i] = np.random.randn(nX, nH)#asignacion de valores aleatorios, deacuerdo al numero de entradas y n. por capa oculta
            b[i] = np.zeros((nH, 1))#creamos un vector de ceros
        elif i != nL + 1: #Capas intermedias
            W[i] = np.random.randn(nH, nH)#asignacion de valores aleatorios entre neuronas de las capas ocultas
            b[i] = np.zeros((nH, 1))#creamos un vector de ceros
        if i == nL:#Ultima capa
            W[i + 1] = np.random.randn(nH, nY)#asignacion de valores aleatorios entre n. capa oculta y capa de salida
            b[i + 1] = np.zeros((nY, 1))#creamos un vector de ceros para el bias

    return (W, b)





#forward propagation
def forwardPropagation(X, Y, W, b):
    m = X.shape[1]
    for i in range(1, nL + 1):
        if i == 1:
            Z[i] = np.dot(W[i].T, X) + b[i]#producto punto entre el peso y la entrada, se le suma el bias
            A[i] = sigmoid(Z[i])#Activacion en ese instante de ciclo
            #A[i] = Hiperbolica(Z[i])
            
        elif i != nL + 1:
            Z[i] = np.dot(W[i], A[i - 1]) + b[i]#producto punto entre la activacion anterior y el peso actual
            A[i] = sigmoid(Z[i])
            #A[i] = Hiperbolica(Z[i])
        if i == nL:
            Z[i + 1] = np.dot(W[i + 1].T, A[i]) + b[i + 1]#producto punto entre la activacion y el peso
            A[i + 1] = sigmoid(Z[i + 1])#activacion final




    #Calculo de la perdida
    logprobs = np.multiply(np.log(A[nL + 1]), Y) + np.multiply(np.log(1 - A[nL + 1]), (1 - Y))
    cost = -np.sum(logprobs) / m
    print("epoch",cost)

    return cost

#BackPropagation
''' en esta seccion realizaremos el calculo del gradiente '''
def backwardPropagation(X, Y, W, A, b):
    m = X.shape[1]#numero de filas en la entrada X
    nL_Final = nL + 1
    for i in range(nL_Final, 0, -1):#de adelante hacia atras
        if i == nL_Final:#capa de salida
            dZ[i] = A[i] - Y#derivada entre la activacion y la capa de salida
            dW[i] = np.dot(dZ[i], A[i - 1].T) / m
            db[i] = np.sum(dZ[i], axis=1, keepdims=True)

        if i < nL_Final and i != 1:#Capas intermedias
            dA[i] = np.dot(W[i + 1], dZ[i + 1])
            dZ[i] = np.multiply(dA[i], A[i] * (1 - A[i]))
            dW[i] = np.dot(dZ[i], A[i - 1].T) / m
            db[i] = np.sum(dZ[i], axis=1, keepdims=True)

        if i == 1:#primera capa
            dA[i] = np.dot(W[i + 1], dZ[i + 1])
            dZ[i] = np.multiply(dA[i], A[i] * (1 - A[i]))
            dW[i] = np.dot(dZ[i], X.T) / m
            db[i] = np.sum(dZ[i], axis=1, keepdims=True)
    return


def updateParameters(W, b, dW, db, learningRate):#Actualizacion de parametros
    nL_Final = nL + 1
    for i in range(1, nL_Final):
        W[i] = W[i] - learningRate * dW[i].T
        b[i] = b[i] - learningRate * db[i]

    return


(W, b) = initializeParameters(nX, nH, nY, nL)

losses = np.zeros((epoch, 1))

#Entrenamiento de la red
for i in range(epoch):
    losses[i, 0] = forwardPropagation(X, Y, W, b)
    backwardPropagation(X, Y, W, A, b)
    updateParameters(W, b, dW, db, learningRate)
    # print ("LOSSES", losses[i, 0])

# Calculo costo y pred5
cost = forwardPropagation(X, Y, W, b)#procentaje de error
pred = (A[nL + 1] > 0.5) * 1.0

#Verificacion
test = open("Real.csv", 'w', newline='')#escribimos en un archivo csv los datos generados
writer = csv.writer(test, dialect='excel')
for item in pred:
    writer.writerow(item)#escribira en el archivo Real.csv los datos calculados por el algoritmo

test1 = open("Deseada.csv", 'w', newline='')
writer1 = csv.writer(test1, dialect='excel')
for item in Y:
    writer1.writerow(item)#Escribira en el archivo los datos de salida


#Grafica
#print(A[nL+1])
print("costo: " + str(cost))
#print("Deseada: "+ str(Y))
#print("Real: "+ str(pred))
plt.figure()
plt.plot(losses)
plt.show()