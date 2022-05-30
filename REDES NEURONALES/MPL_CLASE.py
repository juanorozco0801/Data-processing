import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Creación de la función para la inicialización de la estructura de la red neuronal
# como entradas se tiene: cantidad de neuronas en la capa de entrada (características) -> nX
# cantidad de neuronas en la capa oculta -> nH
# Cantidad de neuronas en la capa de salida (Clases) -> nY
def initializeParameters(nX, nY, nH):
    # Creación de los arreglos
    W1 = np.random.randn(nH, nX)
    b1 = np.zeros((nH, 1))
    W2 = np.random.randn(nY, nH)
    b2 = np.zeros((nY, 1))
    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    return parameters

def forwardPropagation(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    # Cálculo de la perdida siguindo la ecuación de entropia cruzada
    #logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    #cost = -np.sum(logprobs) / m
    #scce = SparseCategoricalCrossentropy()
    #y_true = Y.flatten()
    #y_pred = A2.T
    #cost = scce(y_true, y_pred).numpy()
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m
    print( "este es el costo we ", cost)
    return cost, cache, A2

def backwardPropagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    dZ2 = A2 - Y # Derivada de las activaciones en la neurona de salida
    dW2 = np.dot(dZ2, A1.T) / m # derivada de los pesos W2 con respecto a las activaciones A1
    db2 = np.sum(dZ2, axis=1, keepdims=True) # derivada del bias b2
    
    dA1 = np.dot(W2.T, dZ2) # derivada de las activaciones A1 con respecto a los pesos W2
    dZ1 = np.multiply(dA1, A1*(1 - A1)) # derivada de la función de activación de A1
    dW1 = np.dot(dZ1, X.T) / m # derivada de los pesos W1 con respecto a las entradas X
    db1 = np.sum(dZ1, axis=1, keepdims=True) # derivada del bias b1
    
    gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def updateParameters(parameters, grads, learningRate):
    parameters["W1"] = parameters["W1"] - learningRate*grads["dW1"]
    parameters["W2"] = parameters["W2"] - learningRate*grads["dW2"]
    parameters["b1"] = parameters["b1"] - learningRate*grads["db1"]
    parameters["b2"] = parameters["b2"] - learningRate*grads["db2"]
    return parameters

# Datos de entrada: características y etiquetas
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
# XOR

#Definición de las dimensiones de la red
# Ejemplo de como ingresar datos en la terminal con la función input
#a = input('Ingrese un dato: ')
#os.system("pause")
#print(a)
nH = 2
nX = X.shape[0]
nY = 1

# Definición de parámetros de operación e inicialización de la estructura de la red
parameters = initializeParameters(nX, nY, nH)
numIterations = input('Ingrese la cantidad de iteraciones: ') # Cantidad de epocas o iteraciones del procedimiento de entrenamiento
numIterations = int(numIterations)
learningRate = 0.5 # Taza de aprendizaje
losses = np.zeros((numIterations, 1))

# Entrenamiento de la red
for i in range(numIterations):
    losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
    grads = backwardPropagation(X, Y, cache)
    parameters = updateParameters(parameters, grads, learningRate)

cost, _, A2 = forwardPropagation(X, Y, parameters)
pred = (A2 > 0.5 )*1.0
print(A2)
print(pred)
plt.figure()
plt.plot(losses)
plt.show()


