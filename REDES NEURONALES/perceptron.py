import numpy as np #importamos numpy para usar arreglos de datos
from random import choice  #
from numpy import array, dot, random, sum #importar algunas operaciones con numeros

# Creamos el arreglo dónde se encuentra la base de datos
trainingData = [(array([4,1,1,1,2,1,1,1,1]), 0),
                (array([4,1,3,3,2,1,1,1,1]), 0),
                (array([5,1,1,1,2,1,1,1,1]), 0),
                (array([5,2,2,4,2,4,1,1,1]), 0),
                (array([1,1,1,3,2,3,1,1,1]), 0),
                (array([1,1,1,1,2,2,1,1,1]), 0),
                (array([5,1,1,6,3,1,2,1,1]), 0),
                (array([2,1,1,1,2,1,1,1,1]), 0),
                (array([1,1,1,1,2,1,1,1,1]), 0),
                (array([5,1,1,1,2,1,1,1,1]), 0),
                (array([5,10,8,10,8,10,3,6,3]), 1),
                (array([9,5,5,4,4,5,4,3,3]), 1),
                (array([3,4,5,2,6,8,4,1,1]), 1),
                (array([8,8,7,4,10,10,7,8,7]), 1),
                (array([7,2,4,1,6,10,5,4,3]), 1),
                (array([10,10,8,6,4,5,8,10,1]), 1),
                (array([5,5,5,6,3,10,3,1,1]), 1),
                (array([9,9,10,3,6,10,7,10,6]), 1),
                (array([10,7,7,4,5,10,5,7,2]), 1),
                (array([5,6,7,8,8,10,3,10,3]), 1)]

w = random.rand(9)
Wbias=random.rand(1)
error = []
errors =[]
n = 1000
bias=1
unitStep= lambda x: 0 if x<0 else 1

#def uniStep (x):
#    if x<0:
#        return 0       
#    else:
#        return 1      


#ciclo de entrenamiento


for i in range (n):
    
    x, expected =choice(trainingData) # se selecciona un vector de caraceristicas del arreglo de datos 
    
    resBias=bias*Wbias
    result = dot(w,x)+resBias #realiza el producto punto entre el arreglo en la posicion # y el peso aleatorio que se le asigno
    neuronAct = unitStep(result) #realizo la activacion de la neurona a partir de este caso la funcion escalon, donde si es mayor a cero me retorna 1, de lo contrario un 0
    error = expected - neuronAct # se caulcula el error de predicción
    
    errors.append(error) # se agrega el error a una lista de errores
    w = w + error*x  # se cambia el valor de los pesos calculados inicialmente
    Wbias= Wbias + error*bias 
    #w+=error*x
    pass


for x,_ in trainingData:
    result= dot(w,x)
    print("{}: {} -> {}".format(x[:9], _ ,unitStep(result)))
    pass                          
    
                       
    

