import numpy as np

dataset = np.genfromtxt("dataset9.txt", delimiter=",")
np.random.shuffle(dataset)    
print(dataset)

X = dataset[:,0:7]
Y = dataset[:,7]


nCO = input('Ingrese el numero de capas ocultas: ')               #cantidad de capas ocultas
nCO = int(nCO)
nH = input('Ingrese el numero de neuronas de la capa oculta: ')   #cantidad de neuronas capa oculta
nH = int(nH)

nX = X.shape[1]     #cantidad de neuronas capa entrada
nY = 1              #cantidad de neuronas capa de salida

print(nX)

    
for i in range(5-1,-1,-1):
    
    if i == -0:
        print("capas ",i+5, " UNO ")
       
    elif 0 < i <= 3:
        print("capa ",i, " DOS ")
        
    else:
        print("capa ",i, " TRES ")
            






