#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:37:02 2023

@author: Ricardo de Leon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Kmeans.xlsx', sheet_name='Datos')

X = np.array(df.iloc[:,0])
Y = np.array(df.iloc[:,1])

data = np.asarray(df)

data_max = np.max(data)
data_min = np.min(data)

# Ejercicio 1

centros = []

def get_medias(distancias):
    medias = []
    for i in range(len(df.groupby('asignacion'))):
        media_x = df.loc[df['asignacion']== i, 'x1'].mean()
        media_y = df.loc[df['asignacion']== i, 'x2'].mean()
        medias.append([media_x, media_y])
    return medias

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

# Linea de Prueba
#centros = [[80,80], [90,90]]

print('Centros iniciales: ')
for i in range(len(centros)):
    print(f'C{i} {centros[i]}')

counter = 0

while True:
    
    counter += 1
    # Medimos distancias al centro
    dist = get_distancias(X, Y, centros)
    
    # Clasificamos
    asignaciones = np.argmin(dist,axis=1)
    df['asignacion'] = asignaciones

    # Obtenemos Medias
    NCS = get_medias(dist)
    
    # Condicion para parar si 
    if centros == NCS:
        print(f'Centros Finales encontrados en la iteracion {counter}: ')
        for i in range(len(centros)):
            print(f'C{i} {centros[i]}')
        plt.scatter(X,Y, c='b')
        plt.scatter(*zip(*centros), c='r')
        break

    centros = NCS

# Ejercicio 2
    
def clasifica(x,y):
    print(f'Coordenada ({x},{y}) clasificada como C{np.argmin(get_distancias(x, y, NCS),axis=0)}')

def devuelve_clase():
    print('Clasifica Nueva Coordenada')
    nuevo_x = int(input('Digite Coordenada en X: '))
    nuevo_y = int(input('Digite Coordenada en Y: '))
    clasifica(nuevo_x, nuevo_y)
    
devuelve_clase()