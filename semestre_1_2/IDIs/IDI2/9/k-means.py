#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:48:03 2023

@author: gilberto
"""

import numpy as np

def calcula_centroides(conjunto, k = 10):
    #Obtiene el rango de valores
    min_x, min_y = np.min(conjunto, axis=0)
    max_x, max_y = np.max(conjunto, axis=0)
    #Genera k centroides aleatorios
    nuevos_centroides = [[np.random.uniform(min_x,max_x),
                            np.random.uniform(min_y,max_y)] for i in range(k)]
    
    print("Centroides iniciales")
    print(nuevos_centroides)
    #nuevos_centroides = [[1000,70],[65,78]] #Para probar
    iteraciones = 0
    centroides = []
    while centroides != nuevos_centroides:
        iteraciones += 1
        #Toma los centroides de la iteración pasada
        centroides = nuevos_centroides
        #Calcula la distancia a cada centroide
        distancias = []
        for cent in centroides:
            distancia_cent = (conjunto[:,0] - cent[0])**2+(conjunto[:,1] - cent[1])**2
            distancias.append(distancia_cent)
        print(distancias)
        #Clasifica los puntos de acuerdo a la distancia con los centroides
        clasificaciones = np.argmin(distancias,axis=0)
        break
        #Obtiene los nuevos centroides
        nuevos_centroides = []
        for i in range(k):
            nueva_x = np.mean(conjunto[:, 0], where=clasificaciones==i)
            nueva_y = np.mean(conjunto[:, 1], where=clasificaciones==i)
            #Contempla el caso de que ningún punto quede en este centroide
            if np.isnan(nueva_x) or np.isnan(nueva_y):
                nuevos_centroides.append(centroides[i]) #usa el centroide anterior
            else:    
                nuevos_centroides.append([nueva_x,nueva_y]) #agrega el nuevo
    
    print(f"Centroides finales {iteraciones} iteraciones")
    print(nuevos_centroides)
    return centroides

def clasifica_punto(punto, centroides):
    #Calcula la distancia a cada centroide
    distancias = []
    for cent in centroides:
        distancia_cent = (punto[0] - cent[0])**2+(punto[1] - cent[1])**2
        distancias.append(distancia_cent)
    #Clasifica el puntos de acuerdo a la distancia con los centroides
    return np.argmin(distancias)


################
#Datos de prueba
conjunto_prueba = np.array([[93,43,88,14,66,81,29,100,54,98,11,64,89,81,62,36,15,52,79,45],
                   [96,60,94,25,91,61,95, 86,64,72,19, 3,35, 9,89,22,34,53,66,61]]).T

centroides_calculados = calcula_centroides(conjunto_prueba)
clasificacion = clasifica_punto([70,70], centroides_calculados)
