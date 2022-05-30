from numpy.random import rand
from numpy.random import randint
 
# Función objetivo
#f(x^2, x^2) = 0.0
def objective(x):
	return x[0]**2.0 + x[1]**2.0
 
# Decodificar la cadena de bits a números
def decode(bounds, nBits, bitstring):
	decoded = list()
	largest = 2**nBits
	for i in range(len(bounds)):
		# Extraer la subcadena
		start, end = i * nBits, (i * nBits)+nBits
		substring = bitstring[start:end]
		# Convertir la subcadena a un string
		chars = ''.join([str(s) for s in substring])
		# Convertir el string a integer
		integer = int(chars, 2)
		# Escalar los interos al rango de valores deseado
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# Almacenarlos en la lista
		decoded.append(value)
	return decoded
 
# Selección de torneo
def selection(pop, scores, k=3):
	# Primera selección aleatoria
	selectionIx = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# Verificar si es el mejor (ej realiar un torneo)
		if scores[ix] < scores[selectionIx]:
			selectionIx = ix
	return pop[selectionIx]
 
# Cruzar dos padres para crear dos hijos
def crossover(p1, p2, rCross):
	# Los hijos serán copias por defecto de los padres
	c1, c2 = p1.copy(), p2.copy()
	# Hacer la recombinación
	if rand() < rCross:
		# Seleccionar un punto de cruce que no sea el final del string
		pt = randint(1, len(p1)-2)
		# Hacer el cruce
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 
# Operador de mutación
def mutation(bitstring, rMut):
	for i in range(len(bitstring)):
		# Verifiar la mutación
		if rand() < rMut:
			# Cambiar el bit
			bitstring[i] = 1 - bitstring[i]
 
# Algoritmo genético
def geneticAlgorithm(objective, bounds, nBits, nIter, nPop, rCross, rMut):
	# Generar la población inicial de una cadena de bits aleatoria
	pop = [randint(0, 2, nBits*len(bounds)).tolist() for _ in range(nPop)]
	# Verificar la mejor solución
	best, bestEval = 0, objective(decode(bounds, nBits, pop[0]))
	# Enumerar las generacions
	for gen in range(nIter):
		# Decodificar la población
		decoded = [decode(bounds, nBits, p) for p in pop]
		# Evaluar todos los candidatos en la población
		scores = [objective(d) for d in decoded]
		# Verificar la mejor solución
		for i in range(nPop):
			if scores[i] < bestEval:
				best, bestEval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# Seleccionar los padres
		selected = [selection(pop, scores) for _ in range(nPop)]
		# Crear la siguiente generación
		children = list()
		for i in range(0, nPop, 2):
			# Obtener los padres seleccionados en pares
			p1, p2 = selected[i], selected[i+1]
			# Cruce y mutación
			for c in crossover(p1, p2, rCross):
				# Mutacion
				mutation(c, rMut)
				# Guardar la siguiente generación
				children.append(c)
		# Reemplazar la población
		pop = children
	return [best, bestEval]
 
# Definir el rango de la entrada
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
# Definir el total de iteraciones
nIter = 100
# Bits por variable
nBits = 16
# Definir el tamaño de la población
nPop = 100
# Índice de cruce
rCross = 0.9
# Índice de mutación
rMut = 1.0 / (float(nBits) * len(bounds))
# Aplicar el algoritmo genético
best, score = geneticAlgorithm(objective, bounds, nBits, nIter, nPop, rCross, rMut)
print('Done!')
decoded = decode(bounds, nBits, best)
print(best)
print('f(%s) = %f' % (decoded, score))