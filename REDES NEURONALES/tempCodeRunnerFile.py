import numpy as np


def Hiperbolica(z):
    result = ((2)/(1+np.exp(-2*z)))-1
    return result
    
    
for i in range(5-1,-1,-1):
    
    if i == -1:
        print("capas ",i+5, " UNO ")
       
    elif 0 < i <= 3:
        print("capa ",i, " DOS ")
        
    else:
        print("capa ",i, " TRES ")
            
