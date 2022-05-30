
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import random
import datetime
import statistics as stats
#dataset = np.loadtxt("./data.txt", delimiter=',')
dataset = genfromtxt("dataset9.txt", delimiter=',')
#------------ Arreglo de datos -----------------------
X = (np.array([dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3], dataset[:, 4], dataset[:, 5], dataset[:,6]])) #arreglo de datos de entrada
Y = np.array([dataset[:, 7]]) #Arreglo de datos de salida
nX = X.shape[0] #numero de caracteristicas
nY = Y.shape[0] #numero de salidas
#---------------------- Suministro de datos por el usuario ---------------
nL = int(input('Ingrese numero de capas ocultas: '))
nH = int(input('Ingrese numero de neuronas por capa Oculta: '))
epoch = int(input('Ingrese numero de Iteraciones: '))
learningRate = float(input('Ingrese tasa de aprendizaje: '))
#--------------- Inicializacion de parametros para la red neuronal ------------------
W = {}
b = {}
A = {}
Z = {}
dZ = {}
dW = {}
db = {}
dA = {}
grads = {}

geneSet = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.001,0.4,0.9,0.43,0.1,0.2,0.5] 

#---------- Funcion de activacion Relu --------------------
# Creación de la función de activación relu 0 para z<=0 1 para z>1
def Relu(z):
        if z==0 or z<0 :
            z=0
        else:
            z=1
        return z
# ------- Función de activación sigmoide 1/(1+(e^-z)) ---------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ------- Funcion de activacion Tangente hiperbolica ----------------------
def Hiperbolica(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# ------- Funcion de activacion ReLU ----------------------
def ReLU(z):
    print(z)
    if z < 0:
        z2 = 0
    elif z >= 0:
        z2 = z
    return z2
#-----------Funcion de activacion Softmax --------------
def SoftMax(z):
    return softmax(z)
#---------- Inicializacion de parametros -----------
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


# ------------- forward propagation -----------
def forwardPropagation(X, Y, W, b):
    m = X.shape[1]
    for i in range(1, nL + 1):
        if i == 1:
            Z[i] = np.dot(W[i].T, X) + b[i]#producto punto entre el peso y la entrada, se le suma el bias
            A[i] = sigmoid(Z[i])#Activacion en ese instante de ciclo
            # A[i] = Hiperbolica(Z[i])
            # A[i] = Relu(Z[i])
            # A[i] = SoftMax(Z[i])
        elif i != nL + 1:
            Z[i] = np.dot(W[i], A[i - 1]) + b[i]#producto punto entre la activacion anterior y el peso actual
            A[i] = sigmoid(Z[i])
            # A[i] = Hiperbolica(Z[i])
            #A[i] = Relu(Z[i])
            # A[i] = SoftMax(Z[i])
        if i == nL:
            Z[i + 1] = np.dot(W[i + 1].T, A[i]) + b[i + 1]#producto punto entre la activacion y el peso
            A[i + 1] = sigmoid(Z[i + 1])#activacion final
    #Calculo de la perdida
    logprobs = np.multiply(np.log(A[nL + 1]), Y) + np.multiply(np.log(1 - A[nL + 1]), (1 - Y))
    cost = -np.sum(logprobs) / m

    return cost

#----------- BackPropagation -------------------

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



genes = []

def generateParent(length): #Mejor generación 
    
    while len(genes) < length: # Mientras 0 sea menor a 1 ejecutese 
        sampleSize = min(length - len(genes), len(geneSet)) # Determina iteracion a iteracion cuantos caracteres seleccionar
        genes.extend(random.sample(geneSet, sampleSize)) # Selecciona genes del conjunto de genes y concatena
    return genes[0] #Guarda valor en la primera posición de Genes 

# Función clasificadora de eficiencia del genes para saber si se acerca al objetivo
def fitness(bestParent,CantidadInter,iniI):
    for i  in range(iniI,CantidadInter):
        losses[i, 0]= forwardPropagation(X, Y, W, b)
        backwardPropagation(X, Y, W, A, b)
        updateParameters(W, b, dW, db, bestParent)

    costs = losses[i,0]
    return costs

def mutate(parent):
    index = random.randrange(0, len(geneSet)) # Volver a generar un cambio entre valores del genset
    childGenes = parent # Convierte la información almacenada en padre en una lista
    newGene, alternate = random.sample(geneSet, 2) # Crea un nuevo gen y uno alternado

    if childGenes != alternate:
     newchild = alternate
    else:  
        newchild = newGene
    return newchild

def display(MejorParent,CantidadI,MejorCost, flag):
    if flag == True:
        timeDiff = datetime.datetime.now() - startTime 
        print("Realice :", epoch , " iteraciones", end="\n")
        print("Mejor Padre :", bestParent , end="\n")
        print("Error mas bajo alcanzado con este padre fue: ", MejorCost ,end="\n")
        print("tiempo", str(timeDiff)) 
    else: 
        timeDiff = datetime.datetime.now() - startTime 
        print("Realice :",CantidadI-1, " iteraciones", end="\n")
        print("Mejor Padre :", bestParent ,end="\n")
        print("Error mas bajo alcanzado con este padre fue: ", MejorCost ,end="\n")
        print("tiempo", str(timeDiff))

Prueba = 0.0
losses = np.zeros((epoch, 1))
target=0.05
flag = False
count = 0
CantidadInter = 30
FirsCiclo = 0
iniI= FirsCiclo
Avance = 29
CantidadI = 30
random.seed() # inicilaizacion del algoritmo genetico y genera valores entre los genes y el target
startTime = datetime.datetime.now()
bestParent = generateParent(1) # Se pone en 1 ya qué el valor deseado solo es 1 espacio de memoria
bestFitness = fitness(bestParent,CantidadInter,iniI) # Clasifica al mejor padre

FirsCiclo = FirsCiclo + CantidadInter
iniI = FirsCiclo
CantidadI = CantidadI + CantidadInter


while True:
    Prueba = np.mean(losses)
    target=0.05
    count = count + 1
    child = mutate(bestParent) # Creación de nuevo hijo , nueva mutación
    
    childFitness = fitness(child,CantidadI,iniI)
    
    FirsCiclo = FirsCiclo + CantidadInter
    iniI = FirsCiclo
    CantidadI = CantidadI + CantidadInter
    
    if CantidadI <= epoch:
        if Prueba < 0.001:
            if childFitness < bestFitness:
                bestParent = child
                bestFitness = childFitness

    if childFitness > target and epoch > CantidadI:

        continue
    
    if epoch < CantidadI:
        flag = True
        display(bestParent,CantidadI,bestFitness,flag)
        

    if childFitness <= target:
        flag = False
        display(bestParent,CantidadI,bestFitness,flag)
        break

    if flag == True:
        break
    
    

    
cost = forwardPropagation(X, Y, W, b)
pred = (A[nL+1] > 0.5) * 1.0

print('Total de generaciones: ', CantidadI)


#print(Y)
#print(pred)
plt.figure()
plt.plot(losses)
plt.show()
