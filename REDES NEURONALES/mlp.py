import numpy as np
from matplotlib import pyplot as plt

#datos de entrada: caracteristicas y etiquetas

def initializeParameters(pnH,pnX,pnY):
    
    w1 = np.random.randn(pnH,pnX)
    wb1 = np.random.randn(pnH,1)
    w2 = np.random.randn(pnY,pnH)
    wb2 = np.random.randn(pnY,1)
    
    fncParameters = {"w1" : w1, "wb1" : wb1,
                     "w2" : w2, "wb2" : wb2}    
    
    return fncParameters

def sigmoid(Z_):
    return 1/(1+np.exp(-Z_))


def fordwardPropagation(X_ , Y_, parametersF):
    
    m = X.shape[1]  #cantidad de caracteristicas
    W1 = parametersF["w1"]
    W2 = parametersF["w2"]
    Wb1 = parametersF["wb1"]
    Wb2 = parametersF["wb2"]
    
    Z1 = np.dot(W1,X_)+Wb1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1)+Wb2
    A2 = sigmoid(Z2)
    cachef = (Z1,A1,W1,Wb1,Z2,A2,W2,Wb2)
    
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m
    
    return cost, cachef
    
def backwardPropagation(xB,yB, cacheB):
    
    m = xB.shape[1]
    (Z1, A1, Wb1, Z2, A2, W2, Wb2) = cacheB
    dZ2 = A2 - yB   #derivada de las activaciones en la neurona de salida 
    dW2 = np.dot(dZ2, A1.T)/m # derivada de los pesos respecto a la activacion de la capa oculta
    
    
    
    pass

X = np.array([[0,0,1,1],[0,1.0,1],[0,0,1,1],[0,1,0,1]])
Y = np.array([[0,1,1,0]])


nH = 2                    #cantidad de neuronas ocultas
nX = X.shape[0]          #cantidad de neuronas capa de entrada (número de caracteristicas)
nY = 1                    #cantida de neuronas de salidas 

parameters = initializeParameters(nH,nX,nY)

numIterations = int(input('Ingrese la cantidad de iteraciones en la red'))
learningRate = 1 #taza de aprendizaje
losses = np.zeros((numIterations,1))

for i in range(numIterations):
    losses[i, 0], cache = fordwardPropagation(X, Y, parameters)
    backwardPropagation(X,Y, cache)
    