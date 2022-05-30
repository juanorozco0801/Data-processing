import random # Se importa la libreria random para la generacion de valores aleatorios en la mutacion
import datetime # Se importa la libreria datetime para la medicion de tiempo de ejecucion de un proceso

geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.0123456789'óḿñ"  # Conjunto de genes
target = "todos.van.a.ganar.esta.materia.guiño.guiño" #Objetivo

#  Se crea la funcion padre
def generateParent(length): # La funcion toma como parametro la longitud de la palabra objetivo
    genes = [] # Se crea una lista vacia para el almacenamiento de los genes
    while len(genes) < length:  # Se ejecutan las operaciones mientras la longitud de la lista creada anteriormeente
        # sea menor a la longitud de la palabra objetivo
        sampleSize = min(length - len(genes), len(geneSet)) # Calcula el tamaño del muestreo
        # esta depende de la longitud dada a la funcion, la longitud de los genes usados,
        # y la longitud del conjunto de genes, este valor cambiara en cada iteracion
        genes.extend(random.sample(geneSet, sampleSize)) # como parametros selecciona una cantidad
        # de genes del conjunto de genes creados, la cantidad de genes dependera
        # del tamaño de muestreo, estos valores son concatenados (agregados) a la lista de genes usados
    return ''.join(genes) # Se retornan los elementos que se encuentran dentro de la lista genes y
    # son combinados en una cadena de caracteres
    
def getFitness(guess):
    return sum(1 for expected, actual in zip(target, guess)
               if expected == actual)
    # La funcion retorna la suma de la cantidad de elementos que se encuentran en dos listas a la vez
    # la condicion zip me permite iterar sobre dos listas de manera simultanea
    # lo que hace es evaluar el contenido de las dos listas hasta que el valor esperado sea igual valor actual

def display(parent, time):
    timeDiff = datetime.datetime.now() - time # Calcula el tiempo que le atomado al codigo ejecutarse
    fitness = getFitness(parent) # Evalua en la funcion fittnes la prediccion
    print("{0}\t{1}\t{2}".format(parent, fitness, str(timeDiff))) # Imprime la prediccion, el resultado de
    # fitness y el tiempo que lle ha tomado realizar el proceso

def mutate(parent):
    index = random.randrange(0, len(parent)) # Se determina un índice de manera aleatoria en un rango
    # que va desde 0 hasta la longitud total del padre que es el parametro que recibe la funcion
    childGenes = list(parent) # Convierte la información almacenada en padre en una lista
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate \
        if newGene == childGenes[index] \
        else newGene
    
    '''
    if newGene == childGenes[index]:
        childGenes[index] = alternate
    else:
        childGenes[index] = newGene
    '''
        
    # Se igualan los genes hijos en la posicion indice al alternado
    # Evalua si el nuevo gen es gual al gen hijo en esa posicion
    # De lo contrario continua con el valor el gen
    return ''.join(childGenes) # Se retornan los elementos que se encuentran dentro de la lista genes
    #  hijos y son combinados en una cadena de caracteres separada por comas

random.seed() # Crea la semilla para garantizar que cada vez que se ejecute la funcion aleatorioa
# realmente sea aleatoria
startTime = datetime.datetime.now() # Inicia el conteo de tiempo de ejecucion del proceso
bestParent = generateParent(len(target)) # Crea el tamaño del mejor padre que finalmente tendra
# la misma longitud en caracteres que el objetivo
bestFitness = getFitness(bestParent)
display(bestParent, startTime)

# Evaluar:
count = 0

while True:
    count += 1
    #print(count)
    child = mutate(bestParent) # Indica que los genes hijos van a mutar con referencia al padre
    childFitness = getFitness(child) # Aplica el fitness alos genes hijos
    if bestFitness >= childFitness: # Si el valor del fitness deseado es mayor a el de los hijos que continue operando
       continue
    display(child, startTime) # Muestra los genes hijos
    if childFitness == len(target): # Si el fitness de los hijos es mayor o igual al deseado que termine
        # Esta es la condicion que determina que mi operacion a terminado, el codigo va a ejecutarse
        # evaluando si los genes hijos han mutado a la condicion deseada
        break
    bestFitness = childFitness # Finalmente el mejor resultado va a ser igual al fitness del gen hijo final
    bestParent = child # Y el mejor padre sera eje hijo que produjo el mejor resultado en fitness
    
print('Total de generaciones: ', count)



