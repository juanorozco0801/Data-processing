import numpy as np
from numpy import genfromtxt # Librería para leer el archivo csv
from matplotlib import pyplot as plt
import csv
from numpy.lib.function_base import gradient

dataset = np.genfromtxt("dataset9.txt", delimiter=",")
print(dataset)
np.random.shuffle(dataset)
print(dataset)

def sigmoid(x):
    z = 1 /(1 + np.exp(-x))    
    return z 

# Creación de la función para la inicialización de la estructura de la red neuronal
# como entradas se tiene: cantidad de neuronas en la capa de entrada (características) -> nX
# cantidad de neuronas en la capa oculta -> nH
# Cantidad de neuronas en la capa de salida (Clases) -> nY
def initializeParameters():
    # Creación de los arreglos
    parameters = {}
    for i in range(nL+1):
        keyW = 'W' + str(i+1)
        keyb = 'b' + str(i+1)
        if i == 0:
            W = np.random.randn(nH, nX)
            b = np.zeros((nH, 1))
            parameters[keyW] = W
            parameters[keyb] = b
        elif i == nL:
            W = np.random.randn(nY, nH)
            b = np.zeros((nY, 1))
            parameters[keyW] = W
            parameters[keyb] = b
        else:
            W = np.random.randn(nH, nH)
            b = np.zeros((nH, 1))
            parameters[keyW] = W
            parameters[keyb] = b

    return parameters
def forwardPropagation(X, Y, parameters):
    
    cache = {}

    for i in range(nL+1):
         keyW = 'W' + str(i+1)
         keyb = 'b' + str(i+1)
         keyZ = 'Z' + str(i+1)
         keyA = 'A' + str(i+1) 
         
         W = parameters[keyW]
         b = parameters[keyb]  
               
         if i == 0:
            Z = np.dot(W, X) + b
            A = sigmoid(Z)
         elif i == nL:
            Z = np.dot(W, cache['A' + str(i)]) + b
            A = sigmoid(Z)
         else:
            Z = np.dot(W, cache['A' + str(i)]) + b
            A = sigmoid(Z)
             
         cache[keyZ] = Z
         cache[keyA] = A
         cache[keyW] = W
         cache[keyb] = b
          
    # Cálculo de la perdida siguindo la ecuación de entropia cruzada
    m = X.shape[1]
    logprobs = np.multiply(np.log(cache['A' + str(nL+1)]), Y) + np.multiply(np.log(1 - cache['A' + str(nL+1)]), (1 - Y))
    cost = -np.sum(logprobs)/m
    return cost, cache, cache['A' + str(nL+1)]

def backwardPropagation(X, Y, cache):
    
    gradients = {}
    m = X.shape[1]
    
    for i in range(nL,-1,-1):
        
        keydZ = 'dZ' + str(i+1)
        keydW = 'dW' + str(i+1)
        keydb = 'db' + str(i+1)
    
        if i == nL:
            dZ = cache['A' + str(i+1)] - Y
            dW = np.dot(dZ, cache['A' + str(i)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True)  
                                
        elif i == 0:
            dA = np.dot(cache['W' + str(i + 2)].T, gradients['dZ' + str(i + 2)])
            dZ = np.multiply(dA, cache['A' + str(i + 1)]*(1 - cache['A' + str(i + 1)]))
            dW = np.dot(dZ,X.T)
            db = np.sum(dZ, axis=1, keepdims=True)
        else:
            dA = np.dot(cache['W' + str(i + 2)].T, gradients['dZ' + str(i + 2)])
            dZ = np.multiply(dA, cache['A' + str(i + 1)]*(1 - cache['A' + str(i + 1)]))
            dW = np.dot(dZ,cache['A' + str(i)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True)
                      
        gradients[keydZ] = dZ
        gradients[keydW] = dW
        gradients[keydb] = db  

    #gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
     #            "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def updateParameters(parameters, grads):
    
    son = {}
    for i in range(nL+1):
        keyW = 'W' + str(i+1)
        keyb = 'b' + str(i+1)
        keydW = 'dW' + str(i+1)
        keydb = 'db' + str(i+1)
        son[keyW] = parameters[keyW] - learningRate*grads[keydW]
        son[keyb] = parameters[keyb] - learningRate*grads[keydb]
    
    #parameters['W1'] = parameters['W1'] - learningRate*grads['dW1']
    #parameters['W2'] = pa2rameters['W2'] - learningRate*grads['dW2']
    #parameters['b1'] = parameters['b1'] - learningRate*grads['db1']
    #parameters['b2'] = parameters['b2'] - learningRate*grads['db2']
    return son

def mutate():
    nuevosPesos = {}
    for i in range(nL,-1,-1):
       
        keydW = 'dW' + str(i+1)
        keydb = 'db' + str(i+1)
    
        if i == nL:
            W = np.random.randn(nY, nH)
            b = np.random.randn(nY, 1)
            nuevosPesos[keydW] = W
            nuevosPesos[keydb] = b
        elif i == 0:
            W = np.random.randn(nH, nX)
            b = np.random.randn(nH, 1)
            nuevosPesos[keydW] = W
            nuevosPesos[keydb] = b
        else:
            W = np.random.randn(nH, nH)
            b = np.random.randn(nH, 1)
            nuevosPesos[keydW] = W
            nuevosPesos[keydb] = b

    return nuevosPesos

def ordenPorBurbuja(lista):
    for _ in range(1,11):
        for j in range(1,11-1):
            if lista['f' + str(j)] > lista['f' + str(j+1)]:
                aux=lista['f' + str(j)]
                lista['f' + str(j)]=lista['f' + str(j+1)]
                lista['f' + str(j+1)]=aux
                aux=lista['s' + str(j)]
                lista['s' + str(j)]=lista['s' + str(j+1)]
                lista['s' + str(j+1)]=aux

    return(lista) 

#print(dt)
# Datos de entrada: características y etiquetas
X = (np.array([dataset[:, 0], dataset[:, 1], dataset[:, 2], dataset[:, 3], dataset[:, 4], dataset[:, 5], dataset[:, 6]])) #Arreglo de datos de entrada
Y = np.array([dataset[:, 7]])   #Arreglo de datos de salida

print(Y)
# XOR
#Definición de las dimensi1ones de la red
# Ejemplo de como ingresar datos en la terminal con la función input
'''
nH = int(input("Ingrese la cantidad de Neuronas oculta \n"))
nL = int(input("Ingrese la cantidad de Capas ocultas \n"))
numIterations = int(input("Ingrese la cantidad de Interacciones \n"))
learningRate = float(input("Ingrese la taza de aprendizaje \n"))
'''
nH = 10  #Neuronas en la capa oculta
nL = 10  # Capas ocultas
nX = X.shape[0] # Neuronas en la capa de entrada
nY = 1  # Neuronas en la capa de salida1
# Definición de parámetros de operación e inicialización de la estructura de la red
parameters = initializeParameters()
numIterations = 10 # Cantidad de epocas o iteraciones del procedimiento de entrenamiento    /////Nos habia dado un costo de 0.6931/////
learningRate = 0.05 # Tasa de aprendizaje
losses = np.zeros((numIterations, 1))
lossesChild = np.zeros((numIterations, 1))
bestParent = parameters
sons = []
'''
# Entrenamiento de la red
for i in range(numIterations):
    losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
    grads = backwardPropagation(X, Y, cache)
    parameters = updateParameters(parameters, grads)

cost, _, A2 = forwardPropagation(X, Y, parameters)
pred = (A2 > 0.5 )*1.0
#print(A2)
#print(pred)
plt.figure()
plt.plot(losses)
plt.show()
print()
'''
for i in range(numIterations):  
    losses[i,0], cache, A2 = forwardPropagation(X, Y, bestParent)
    fitnessParent = losses[i,0]
    
    hijos = {}
    for i in range(0,10,1):
        keys = 's' + str(i+1)
        keyf = 'f' + str(i+1)
        smallChange = mutate()
        hijos[keys] = updateParameters(bestParent,smallChange)
        lossesChild, cache, A2 = forwardPropagation(X, Y, hijos[keys])
        hijos[keyf] = lossesChild   
    sons = ordenPorBurbuja(hijos)
    
    if fitnessParent <= sons['f1']:
        continue
    bestFitness = sons['f1']
    bestParent = sons['s1']
    print(bestFitness)
   
cost, _, A2 = forwardPropagation(X, Y, bestParent)
pred = (A2 > 0.5 )*1.0
#print(A2)
print(pred)
plt.figure()
plt.plot(losses)
plt.show()
print()
