#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:53:33 2022

@author: Ricardo De Leon
"""

import numpy as np

# Genere una matriz Q de 5x5 con elementos aleatorios reales en [1,10)

Q=np.random.uniform(1,10, size=(5, 5))

# Imprima el porcentaje de elementos de la matriz Q que están por encima del promedio.

print(f'{(np.count_nonzero(Q > np.mean(Q))/np.count_nonzero(Q))*100}%')

# Genere una lista L con 15 listas de 100 números reales en [0,8)

L=[[np.random.uniform(0, 8) for i in range(100)] for j in range(15)]

# Genere una lista M con los promedios de cada una de las 15 listas de L, luego imprima el promedio de los promedios.

M=np.mean(L, axis=1) 
print(np.mean(M))

# Emule el experimento de obtener la suma de dos dados y genere una lista aleatoria D con los resultados de 1000 lanzamientos (note que no todos los resultados son igual de probables).

D=[np.random.randint(1, 7) + np.random.randint(1, 7) for i in range(1000)]
    
# Suponga que tiene una caja con dos canicas rojas, tres azules y 5 blancas. Quiere emular el proceso de sacar 3 canicas de la caja de forma que no se regresa la canica que ya se sacó. Cree una lista C con los resultados de 10 experimentos (es decir, una lista de 10 listas con 3 elementos). 

C=[list(np.random.choice(2 * ['R'] + 3 * ['A'] + 5 * ['B'],3,False)) for i in range(10)]
