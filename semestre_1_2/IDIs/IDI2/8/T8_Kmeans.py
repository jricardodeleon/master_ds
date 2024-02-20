#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:56:30 2023

@author: Ricardo De Leon
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

def get_centros():
    cx = np.random.randint(data_min, data_max)
    cy = np.random.randint(data_min, data_max)
    return np.asarray([cx,cy])

# definimos centros
c1 = get_centros()
c2 = get_centros()

# Descomentar para probar ejercicio en clase
# c1 = np.array([80,80])
# c2 = np.array([90,90])

print(f'Centros iniciales c1 = {c1} , c2 = {c2}')

counter = 0

while True:
    counter += 1
    
    # Medimos distancias a centro
    
    dist_c1 = (X - c1[0]) ** 2 + (Y - c1[1]) ** 2
    dist_c2 = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2
    
    # Clasificamos
    
    asignacion = []
    for i in range(len(dist_c1)):
        if dist_c2[i] <= dist_c1[i]:
            asignacion.append('c2')
        else:
            asignacion.append('c1')
            
    df['asignacion'] = asignacion
    
    # Obtenemos Medias
    
    media_c1_x = df.loc[df['asignacion']=='c1', 'x1'].mean()
    media_c1_y = df.loc[df['asignacion']=='c1', 'x2'].mean()
    media_c2_x = df.loc[df['asignacion']=='c2', 'x1'].mean()
    media_c2_y = df.loc[df['asignacion']=='c2', 'x2'].mean()
    
    nc1 = np.array([media_c1_x, media_c1_y])
    nc2 = np.array([media_c2_x, media_c2_y])
    
    if np.linalg.det([[c1,nc1],[c2,nc2]])[0] == 0.0:
        print(f'Centros finales c1 = {np.round(c1,2)} , c2 = {np.round(c2,2)} iter # {counter}')
        break
    
    c1 = nc1
    c2 = nc2
    print(f'Nuevos centros c1 = {np.round(c1,2)} , c2 = {np.round(c2,2)}')
    
# Ejercicio 2
    
def clasifica(x,y):
    dist_c1 = (x - c1[0]) ** 2 + (y - c1[1]) ** 2
    dist_c2 = (x - c2[0]) ** 2 + (y - c2[1]) ** 2
    if dist_c2 <= dist_c1:
        print(f'Coordenada ({x},{y}) clasificada como C2')
    else:
        print(f'Coordenada ({x},{y}) clasificada como C1')

def devuelve_clase():
    nuevo_x = int(input('Digite Coordenada en X: '))
    nuevo_y = int(input('Digite Coordenada en Y: '))
    clasifica(nuevo_x, nuevo_y)
    
devuelve_clase()