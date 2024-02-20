#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:57:00 2023

@author: gilberto
"""

import numpy as np


def kohonen(conjunto, k = 2, E = 1e-1, maximo_iteraciones=1000, debug=False):
    #Obtiene el rango de valores
    min_x, min_y = np.min(conjunto, axis=0)
    max_x, max_y = np.max(conjunto, axis=0)
    #Genera k centroides aleatorios
    # centroides = np.array([[np.random.uniform(min_x,max_x),
    #                         np.random.uniform(min_y,max_y)] for i in range(k)])
    centroides = np.array([[20.0,50.0],[50.0,50.0]]) #Para probar
    
    print(f"k={k} E={E} Centroides iniciales")
    print(centroides)
    t = 1 #T para calcular alfa
    error = E+t
    while error > E and t < maximo_iteraciones:
        t += 1
        centroides_ant = centroides.copy()
        #Recorre los puntos en carrusel
        for i in range(conjunto.shape[0]):
            #Obtiene el centroide más cercano de acuerdo a la distancia
            j = clasifica_punto(conjunto[i], centroides)
            #Ajusta el centroide
            centroides[j] = 1/t*conjunto[i] + (1-1/t)*centroides[j]
        #Calculo el movimiento de los centroides
        error = np.max(abs(centroides_ant - centroides))
        print(f"Iteración {t-1} Error {error} Centroides") if debug else ""
        print(centroides) if debug else ""
    
    print(f"Iteración {t-1} Error {error} Centroides finales")
    print(centroides)
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

centroides_calculados = kohonen(conjunto_prueba)
clasifica_punto([70,70], centroides_calculados)
