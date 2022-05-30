import random       #se importa la libreria random para genreacion de valores aleatorios
import datetime     #se importa la libreria datetime para la medicion de tiempo de ejecucion

geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.0123456789'ó"
target = "que tranza la garbanza"

def generateParent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet)) # me entrega el valor entre (x,Y)
        genes.extend(random.sample(geneSet, sampleSize)) #de geneset tomo tantos valores random determinados por samplesize
    return ''.join(genes)

def getFitness(guess):
    return sum(1 for expected, actual in zip(target, guess)
                if expected == actual)

def display(parent, time):
    timeDiff = datetime.datetime.now() - time
    fitness = getFitness(parent)
    print("{0}\t{1}\t{2}".format(parent,fitness,str(timeDiff)))

def mutate(parent):
    index = random.randrange(0,len(parent))
    childGenes = list(parent) # se convierte en lista para poder editar sus vsalores, en caso de no hacerlo da error
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate\
        if newGene == childGenes[index]\
        else newGene
    '''
    if newGene == childGenes[index]:
        childGenes[index] = alternate
    else:
        childGenes[index] = newGene
    '''
    return ''.join(childGenes)

random.seed() #genera una semilla para generar valores aleatorios diferentes
startTime = datetime.datetime.now()    # captura la hora del sistema
bestParent = generateParent(len(target)) #se manda tamaño del vector
bestFitness = getFitness(bestParent)
display(bestParent, startTime)

# Evaluar: 
count = 0

while True:
    count += 1
    child = mutate(bestParent)
    childFitness = getFitness(child)
    if bestFitness >= childFitness:
        continue
    display(child, startTime)
    if childFitness == len(target):
        break
    bestFitness = childFitness
    bestParent = child
    
print('total de generaciones: ', count)