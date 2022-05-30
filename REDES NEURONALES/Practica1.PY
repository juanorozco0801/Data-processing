import numpy as np
import random
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Creación de la función para la inicialización de la estructura de la red neuronal
# como entradas se tiene: cantidad de neuronas en la capa de entrada (características) -> nX
# cantidad de neuronas en la capa oculta -> nH
# Cantidad de neuronas en la capa de salida (Clases) -> nY

def initializeParameters(nX_, nY_, nH_, nCO_):
    # Creación de los arreglos
    
    WE = np.random.randn(nH, nX)
    bE = np.random.randn(nH,1)
    
    WCO = np.random.rand(nH,nH,nCO-1)
    bCO = np.zeros((nH,960,nCO-1))
    
    #AC1   AC2   AC3     AC4     AC5
    
        
    WS = np.random.randn(nY, nH)
    bS = np.zeros((nY, 1))
    
    #for i in range(nCO-1):
        #print("capa " ,i, ": ",WCO[:,:,i])
        
        #pass
    
    
    parameters = {"W1": WE,     "b1": bE,
                  "WCO": WCO,   "bCO":bCO,
                  "W2": WS,     "b2": bS}
    return parameters


def forwardPropagation(X, Y, parameters,nCo):
    
    m = X.shape[1]
    
    WE = parameters["W1"]
    WCO = parameters["WCO"]
    WS = parameters["W2"]
    b1 = parameters["b1"]
    bCO = parameters["bCO"]
    b2 = parameters["b2"]
    
    ZCO = np.zeros((nH,960,nCo))
    ACO = np.zeros((nH,960,nCO))
    
    ZCO[:,:,0] = np.dot(WE, X) + b1
    ACO[:,:,0] = sigmoid(ZCO[:,:,0])
    
    #ZCO = np.dot(WCO[:,:,0],A1)

    for i in range(1,nCo):
        
            ZCO[:,:,i] = np.dot(WCO[:,:,i-1],ACO[:,:,i-1]) + bCO[:,:,i-1]
            ACO[:,:,i] = sigmoid(ZCO[:,:,i])
            #print("capa ",i, ": ",ACO[:,:,i])

    pass
    
    Z2 = np.dot(WS, ACO[:,:,nCO-1]) + b2
    A2 = sigmoid(Z2)
    cache = (b1, WE, ZCO, ACO, bCO, WCO, Z2, A2, WS, b2)
    
    # Cálculo de la perdida siguindo la ecuación de entropia cruzada
    #logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    #cost = -np.sum(logprobs) / m
    #scce = SparseCategoricalCrossentropy()
    #y_true = Y.flatten()
    #y_pred = A2.T
    #cost = scce(y_true, y_pred).numpy()
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m
    return cost, cache, A2

def backwardPropagation(X, Y, cache, nCO):
    m = X.shape[1]
    (b1, WE, ZCO, ACO, bCO, WCO, Z2, A2, WS, b2) = cache
    
    dZS = A2 - Y.T # Derivada de las activaciones en la neurona de salida
    dWS = np.dot(dZS, ACO[:,:,nCO-1].T) / m # derivada de los pesos W2 con respecto a las activaciones A1
    dbS = np.sum(dZS, axis=1, keepdims=True) # derivada del bias b2
    
    dACO = np.zeros((nH,960,nCO))
    dWCO = np.zeros((nH,nH,nCO-1))
    dZCO = np.zeros((nH,960,nCO))
    dbco = np.zeros((nH,1,nCO-1))
    
    for i in range(nCO-1,-1,-1):

        if i == nCO-1:

            dACO[:,:,i] = np.dot(WS.T,dZS)
            dZCO[:,:,i] = np.multiply(dACO[:,:,i], ACO[:,:,i]*(1 - ACO[:,:,i]))
            dWCO[:,:,i-1] = np.dot(dZCO[:,:,i], ACO[:,:,i-1].T) / m # derivada de los pesos W1 con respecto a las entradas X
            dbco[:,:,i-1] = np.sum(dZCO[:,:,i], axis=1, keepdims=True) # derivada del bias b1

        elif 0 < i <= 3:
            
            dACO[:,:,i] = np.dot(WCO[:,:,i].T,dZCO[:,:,i+1])
            dZCO[:,:,i] = np.multiply(dACO[:,:,i], ACO[:,:,i]*(1 - ACO[:,:,i]))
            dWCO[:,:,i-1] = np.dot(dZCO[:,:,i], ACO[:,:,i-1].T) / m # derivada de los pesos W1 con respecto a las entradas X
            dbco[:,:,i-1] = np.sum(dZCO[:,:,i], axis=1, keepdims=True) # derivada del bias b1

        else:

            dACO[:,:,i] = np.dot(WCO[:,:,i].T,dZCO[:,:,i+1])
            dZCO[:,:,i] = np.multiply(dACO[:,:,i], ACO[:,:,i]*(1 - ACO[:,:,i]))
            dWE = np.dot(dZCO[:,:,i], X.T) / m # derivada de los pesos W1 con respecto a las entradas X
            db1 = np.sum(dZCO[:,:,i], axis=1, keepdims=True) # derivada del bias b1
            

    
            

    gradients = {"dWS": dWS, "dbS": dbS,
                 "dWCO": dWCO, "dbCO": dbco,
                 "dWE": dWE, "dbE": db1}

    return gradients

def updateParameters(parameters, grads, learningRate):
    
    parameters["W1"] = parameters["W1"] - learningRate*grads["dWE"]
    parameters["WCO"] = parameters["WCO"] - learningRate*grads["dWCO"]
    parameters["W2"] = parameters["W2"] - learningRate*grads["dWS"]
    parameters["b1"] = parameters["b1"] - learningRate*grads["dbE"]
    parameters["bCO"] = parameters["bCO"] - learningRate*grads["dbCO"]
    parameters["b2"] = parameters["b2"] - learningRate*grads["dbS"]
    return parameters

# Datos de entrada: características y etiquetas


dataset = np.genfromtxt("dataset.txt",delimiter=",")    # lectura de dataset
np.random.shuffle(dataset)                              # Reorganiza la base de datos aleatoriamente

X = dataset[:,0:7]
Y = dataset[:,7]
X = X.T
Y = Y.reshape(960,1)


#XOR

#Definición de las dimensiones de la red
# Ejemplo de como ingresar datos en la terminal con la función input
#a = input('Ingrese un dato: ')
#os.system("pause")
#print(a)

#nCO = input('Ingrese el numero de capas ocultas: ')               #cantidad de capas ocultas
nCO = 5
nCO = int(nCO)
#nH = input('Ingrese el numero de neuronas de la capa oculta: ')   #cantidad de neuronas capa oculta
nH = 5
nH = int(nH)


nX = X.shape[0]     #cantidad de neuronas capa entrada
#nY = Y.shape[0]     #cantidad de neuronas capa de salida
nY = 1

# Definición de parámetros de operación e inicialización de la estructura de la red
parameters = initializeParameters(nX, nY, nH, nCO)
#numIterations = input('Ingrese la cantidad de iteraciones: ') # Cantidad de epocas o iteraciones del procedimiento de entrenamiento
numIterations = 100
numIterations = int(numIterations)     #se define como entero
learningRate = 0.5 # Taza de aprendizaje    se deberia definir como float?
losses = np.zeros((numIterations, 1))


# Entrenamiento de la red
for i in range(numIterations):
    
    losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters,nCO)
    grads = backwardPropagation(X, Y, cache,nCO)
    parameters = updateParameters(parameters, grads, learningRate)

cost, _, A2 = forwardPropagation(X, Y, parameters,nCO)
pred = (A2 > 0.5 )*1.0
print(A2)
#print(pred)
plt.figure()
plt.plot(losses)
plt.show()
pass

