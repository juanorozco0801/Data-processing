import numpy as np


def Hiperbolica(z):
    result = ((2)/(1+np.exp(-2*z)))-1
    return result
    
for i in range(-5,10,0.1):

    activacion = Hiperbolica(i)
    print(activacion)
    
    
    
