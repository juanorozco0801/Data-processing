import os

# Grafo
graph = {'ARAD': ['ZERIND', 'TIMISOARA', 'SIBIU'],
         'ZERIND': ['ORADEA', 'ARAD'],
         'TIMISOARA': ['ARAD', 'LUGOJ'],
         'SIBIU': ['ARAD','ORADEA','FAGARAS','RIMNICU VILCEA'],
         'ORADEA': ['ZERIND', 'SIBIU'],
         'LUGOJ': ['TIMISOARA','MEHADIA'],
         'MEHADIA': ['LUGOJ','DOBRETA'],
         'DOBRETA': ['MEHADIA','CRAIOVA'],
         'CRAIOVA': ['DOBRETA','PITESTI','RIMNICU VILCEA'],
         'RIMNICU VILCEA': ['SIBIU','PITESTI','CRAIOVA'],
         'PITESTI': ['RIMNICU VILCEA','CRAIOVA','BUCHAREST'],
         'FAGARAS': ['SIBIU','BUCHAREST'],
         'BUCHAREST': ['PITESTI','FAGARAS','GIURGIU','URZICENI'],
         'GIURGIU': ['BUCHAREST'],
         'URZICENI': ['BUCHAREST','HIRSOVA','VASLUI'],
         'HIRSOVA': ['EFORIE','URZICENI'],
         'EFORIE': ['HIRSOVA'],
         'VASLUI': ['URZICENI','LASI'],
         'LASI': ['VASLUI','NEAMT'],
         'NEAMT': ['LASI'],}

# Exploracion
start1 = input("Ingrese el nodo de inicio: ")

# Se visitan todos los nodos del grafo (componentes conectados)
def bfs_explore(gp, start):
    # Se hace seguimiento a los nodos visitados
    explored = []
    # Se hace seguimiento a los nodos seleccionados
    queue = [start]

    # Itera hasta que no hallan as nodos para seleccionar
    while queue:
        # Elimina el nodo menos profundo (primero nodo) de la lista de espera
        node = queue.pop(0)
        if node not in explored:
            # Agrega el nodo a la lista de nodos seleccionados
            explored.append(node)
            neighbours = gp[node]
            #  Agrega los vecinos del nodo a la lista de espera
            for neighbour in neighbours:
                queue.append(neighbour)
    return explored

result1 = bfs_explore(graph, start1)
print(result1)

os.system("pause")

# Busqueda
start2 = input("Ingrese el nodo de inicio: ")
goal = input("Ingrese el nodo objetivo: ")

# Encuentra el camino mas corto entre dos nodos del grafo
def bfs_search(graph, start, goal):
    # Se hace seguimiento a los nodos visitados
    explored = []
    # Se hace seguimiento a los nodos seleccionados
    queue = [[start]]

    # Regresa el camino si el inicio es la meta
    if start == goal:
        return "El nodo de inicio es el nodo objetivo"

    # Itera hasta que no hallan as nodos para seleccionar
    while queue:
        # Elimina el primero camino de la lista de espera
        path = queue.pop(0)
        #  Toma el ultimo nodo del camino
        node = path[-1]
        if node not in explored:
            neighbours = graph[node]
            # Recorre todos los nodos vecinos, construyendo un nuevo camino y lo agrega a la lista de espera
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # Regresa el camino si un vecino es la meta
                if neighbour == goal:
                    return new_path

            # Marca el nodo como explorado
            explored.append(node)

    # En caso de que no exista un camino entre los dos nodos
    return "No existe un camino entre los dos nodos"


result2 = bfs_search(graph, start2, goal)
print(result2)

# Fuentes:
# https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/
# http://cyluun.github.io/blog/uninformed-search-algorithms-in-python
# https://gist.github.com/eJavierpr0/b0dd8161bdab70f9e9e990c4db5c376a
# http://www.koderdojo.com/blog/depth-first-search-in-python-recursive-and-non-recursive-programming

