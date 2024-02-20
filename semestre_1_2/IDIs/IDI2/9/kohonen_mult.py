#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:46:20 2023

@author: Ricardo De León
"""

import pandas as pd
import numpy as np

df = pd.read_excel('Kmeans.xlsx', sheet_name='Datos')

X = np.array(df.iloc[:,0])
Y = np.array(df.iloc[:,1])

data = np.asarray(df)

data_max = np.max(data)
data_min = np.min(data)

# Ejercicio 1

centros = []
centros_tmp = []


def clasifica(x,y):
   return np.argmin(get_distancias(x, y, centros),axis=0)

def get_distancias(X,Y, c):
    distancias = []
    for i in range(len(c)):
        distancias.append((X - c[i][0]) ** 2 + (Y - c[i][1]) ** 2)
    return np.array(distancias).T

def get_centros():
    cx = np.random.randint(data_min, data_max)
    cy = np.random.randint(data_min, data_max)
    return [cx,cy]

def crear_centros():
    k = int(input("Cuantos centros quieres crear (Minimo 2)? "))
    for _ in range(k):
        centros.append(get_centros())

# Inicializando centros
crear_centros()

# # Linea de Prueba
# centros = [[20,50], [50,50]]


print('Centros iniciales: ')
for i in range(len(centros)):
    print(f'C{i} {centros[i]}')

# definicion de parametros de inicio

E = 1e-1
error = E + 1
centros_tmp = centros.copy()
lerning_rates = 51

for t in range(1,lerning_rates):
    
        # Carrusel
        for i in range(len(X)):
            # Clasifica
            c = clasifica(X[i], Y[i])
            # Recorre centroide
            centros[c] = 1/t * np.array([X[i], Y[i]]) + (1-1/t) * np.array(centros[c])
        
        error = np.max(abs(np.vstack(centros_tmp) - np.vstack(centros)))
        
        # Condicion de paro
        if error < E:
            print(f'Centros Finales encontrados en la iteracion {t-1}: ')
            for i in range(len(centros)):
                print(f'C{i} {centros[i]}')
            break
        centros_tmp = centros.copy()
        
print('Error', error)

# Ejercicio 2

def devuelve_clase():
    print('Clasifica Nueva Coordenada')
    nuevo_x = int(input('Digite Coordenada en X: '))
    nuevo_y = int(input('Digite Coordenada en Y: '))
    c = clasifica(nuevo_x, nuevo_y)
    print(f'Coordenada ({nuevo_x},{nuevo_y}) clasificada como C{c}')
    
devuelve_clase()


