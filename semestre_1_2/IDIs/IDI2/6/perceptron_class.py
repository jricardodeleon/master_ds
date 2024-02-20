#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:33:10 2023

@author: gilberto
"""

import numpy as np

class Perceptron:
    def __init__(self, numEntradas, numSalidas, numNeuronas):
        #inicializa las 
        self.N = numEntradas
        self.M = numSalidas
        self.L = numNeuronas
        #Pesos para la capa oculta
        self.wh = np.random.uniform(-1, 1, (self.L, self.N))
        #Pesos para la capa de salida
        self.wo = np.random.uniform(-1, 1, (self.M, self.L))
        
    def sigmoide(self, x, a=1):
        return 1/(1+np.exp(-a*x))
    
    def forward(self, xj):
        net_h = self.wh@xj
        y_h = self.sigmoide(net_h) #Capa oculta
        net_o = self.wo@y_h
        y = self.sigmoide(net_o) #Capa de salida
        return y_h, y
    
    def backwards(self, dj, y, y_h, xj, alfa):
        d_o = (dj-y)*y*(1-y) #delta minúscula para las salidas
        d_h = y_h*(1-y_h)*(self.wo.T@d_o) #delta minúscula para la capa oculta
        delta_wo = (alfa*d_o)@y_h.T #Delta wo
        delta_wh = (alfa*d_h)@xj.T #Delta wh
        self.wo += delta_wo #Se ajustan pesos para la capa de salida
        self.wh += delta_wh #Se ajustan pesos para la capa oculta
        error = np.max(abs(d_o))
        return error
    
    def entrenamiento(self, x, d, alfa=1e-2, E=1e-3, maximo_iteraciones=100000):
        iteraciones = 0
        error = 1+E
        while error > E and iteraciones < maximo_iteraciones:
            iteraciones += 1
            #Iterar sobre cada observación
            for j in np.arange(len(x)):
                #Renglón j
                xj = np.reshape(x[j], (self.N,1)) #Asegura la dimensión
                dj = np.reshape(d[j], (self.M,1)) #Asegura la dimensión
                y_h, y = self.forward(xj) #Calcula la salida con esas entradas
                error = self.backwards(dj, y, y_h, xj, alfa) #Obtiene ajuste
        print(f"Iteraciones: {iteraciones}, Error: {error}")

"""
Inicio de la ejecución
"""
import pandas as pd
df = pd.read_excel('./tabla_para_probar.xlsx')

#Obtiene los nombres de columnas para entradas y salidas
xcols = []
dcols = []
for c in df.columns:
    if c.startswith('x'):
        xcols.append(c)
    else:
        dcols.append(c)

#Obtiene la matriz con las entradas y la matriz con las salidas para entrenar
x = np.array(df[df['d1'] != '?'][xcols])
d = np.array(df[df['d1'] != '?'][dcols], dtype = int)
#Obtiene las entradas a reconocer
xf = np.array(df[df['d1'] == '?'][xcols])

p1 = Perceptron(len(xcols), len(dcols), len(xcols) + len(dcols))
p1.entrenamiento(x, d, 1e-1)

#Iterar sobre los datos de entrenamiento para ver el resultado
d_estimada = []
for j in range(len(x)):
    #Renglón j
    xj = np.reshape(x[j], (len(x[j]),1)) #Asegura la dimensión
    dj = np.reshape(d[j], (len(d[j]),1)) #Asegura la dimensión
    _, y = p1.forward(xj) #Calcula la salida con esas entradas
    d_estimada.append(y.tolist())
