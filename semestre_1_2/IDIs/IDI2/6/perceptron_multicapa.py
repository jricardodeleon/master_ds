#!/usr/bin/env pDthon3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:11:55 2023

@author: khrw896
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('tabla_para_probar.xlsx', na_values='?')
df_non_na = df[~df.isnull().any(axis=1)]
df_na = df[df.isnull().any(axis=1)]

# Definir entradas
X = np.array(df_non_na[[col for col in df[~df.isnull().any(axis=1)] if col.startswith('x')]])
D = np.array(df_non_na[[col for col in df[~df.isnull().any(axis=1)] if col.startswith('d')]])
X_to_evaluate = np.array(df_na[[col for col in df[df.isnull().any(axis=1)] if col.startswith('x')]])

# Definir los parámetros del modelo en donde L es el numero de neuronas
N = X.shape[1]
M = D.shape[1]
Q = np.array(df_non_na).shape[1]
L = 5
alpha = 0.1
iteraciones = 100000
E = 1e-3

# Definir los pesos para la capa oculta wo D capa de salida wo
wo = np.random.uniform(low=-1, high=1, size=(M, L)) - 0.5
wh = np.random.uniform(low=-1, high=1, size=(L, N)) - 0.5


def forward(X, wo, wh, iteracion):
    neth = wh @ X[iteracion].T
    yh = 1 / (1 + np.exp(-neth))
    neto = wo @ yh
    y = 1 / (1 + np.exp(-neto))
    return yh, y

def backward(D, yh, y, iteracion):
    # del_wh = np.zeros((len(wh), wh.shape[1]))
    # del_wo = np.zeros((len(wo), wo.shape[1]))
    del_wh = np.zeros_like(wh)
    del_wo = np.zeros_like(wo)
    do = (D[iteracion].T - y) * (y * (1 - y))
    dh = yh * ( 1 - yh) * (wo.T @ do)
    del_wo += np.reshape(alpha*do,(M,1)) @ np.reshape(yh.T,(1,L))
    del_wh += np.reshape(alpha*dh,(L,1)) @ np.reshape(X[iteracion],(1,N))
    return dh, do, del_wo, del_wh

errors = []
for i in range(iteraciones):
    for j in range(Q):
        yh, y = forward(X, wo, wh, j)
        _, do, del_wo, del_wh = backward(D, yh, y, j)
        error=np.linalg.norm(do)
        print(error)
        wo+=del_wo
        wh+=del_wh
    if error < E:
        print(f'Limite de error alcanzado en iter {i}')
        break
    errors.append(error)

print('\n Comprobación de datos entrenados')
for i in range(X.shape[1]):
    _, y = forward(X, wo, wh, i)
    print('Datos De Entrada:', X[i], 'Salidas Conocidas', D[i], 'Salidas Entrenadas', y, 'redondeadas', np.round_(y,0))

print('\n Predicción datos para entradas conocidas y salidas desconocidas')
for i in range(X_to_evaluate.shape[1]):
    _, y = forward(X_to_evaluate, wo, wh, i)
    print('Datos De Entrada:', X_to_evaluate[i],'Salidas Entrenadas', y, 'redondeadas', np.round_(y,0))
    
plt.plot(errors)
plt.xlabel("Iteraciones")
plt.ylabel("Errores")
plt.title("Training Errores")
plt.show()
    