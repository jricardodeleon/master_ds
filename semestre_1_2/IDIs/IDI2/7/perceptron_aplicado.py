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
    
    def entrenamiento(self, x, d, alfa=10000, E=1e-3, maximo_iteraciones=1000):
        iteraciones = 0
        error = 1+E
        while error > E and iteraciones < maximo_iteraciones:
            iteraciones += 1
            #Iterar sobre cada observación
            error_j = []
            for j in np.arange(len(x)):
                #Renglón j
                xj = np.reshape(x[j], (self.N,1)) #Asegura la dimensión
                dj = np.reshape(d[j], (self.M,1)) #Asegura la dimensión
                y_h, y = self.forward(xj) #Calcula la salida con esas entradas
                error_j.append(self.backwards(dj, y, y_h, xj, alfa)) #Obtiene ajuste
            #Obtiene el error de entrenar con todos los renglones
            error = np.max(error_j)
            print(f"Iteraciones: {iteraciones}, Error: {error}")
        
    def ejecuta(self, x):
        y = np.zeros((len(x),self.M))
        #Itera por el conjunto de entradas
        for j in np.arange(len(x)):
            #Renglón j
            xj = np.reshape(x[j], (self.N,1)) #Asegura la dimensión
            _, y[j] = self.forward(xj) #Calcula la salida con esas entradas
        return y > 0.5       

"""
Inicio de la ejecución
"""
import pandas as pd
df = pd.read_excel('./PercMultAplicado.xlsx')

#Solo los datos que nos interesan
#Normaliza el monto
df_monto = np.sqrt(df['Monto'])
df_monto_norm = df_monto/(np.max(df_monto)-np.min(df_monto))
df_carga_mensual = df['Monto']/df['Ingreso mensual']  #Carga de la mensualidad al salario
df_antiguedad_norm = df['Antigüedad laboral (meses)']/(np.max(df['Antigüedad laboral (meses)'])-np.min(df['Antigüedad laboral (meses)']))

df_posibilidad_mora = pd.concat([df_monto_norm,
                                 df_carga_mensual,
                                 df_antiguedad_norm,
                                 (df['Mora'] == 'SI').astype(int)], axis=1)

df_posibilidad_mora.columns = ['Monto', 'Carga Mensual', 'Antigüedad', 'Mora']


df_muestra = df_posibilidad_mora.groupby('Mora').apply(lambda x: x.sample(frac=0.70))
x = np.array(df_muestra[['Monto', 'Carga Mensual', 'Antigüedad']])
d = np.array(df_muestra[['Mora']])

p1 = Perceptron(3, 1, 4)
p1.entrenamiento(x, d, 15, maximo_iteraciones=5000)

y = p1.ejecuta(x)
#Medir el accuracy del entrenamiento
comp = (d.astype(bool) == y)
accuracy_training = np.mean(comp)
accuracy_training

#Probar con todas
x_total = np.array(df_posibilidad_mora[['Monto', 'Carga Mensual', 'Antigüedad']])
d_total = np.array(df_posibilidad_mora[['Mora']])
y_total = p1.ejecuta(x_total)
accuracy_total = np.mean(d_total.astype(bool) == y_total)
accuracy_total
