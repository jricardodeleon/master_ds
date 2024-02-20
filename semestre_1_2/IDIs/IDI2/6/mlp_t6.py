#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:58:33 2023

@author: khrw896
"""

import pandas as pd
import numpy as np

df = pd.read_excel('tabla_para_probar.xlsx', na_values='?')
df_non_na = df[~df.isnull().any(axis=1)]
df_na = df[df.isnull().any(axis=1)]

# Definir entradas
X = np.array(df_non_na[[col for col in df[~df.isnull().any(axis=1)] if col.startswith('x')]])
D = np.array(df_non_na[[col for col in df[~df.isnull().any(axis=1)] if col.startswith('d')]])

# Definir los parámetros del modelo en donde L es el numero de neuronas
N = X.shape[1]
M = D.shape[1]
L = 5
alpha = 0.001
iteraciones = 1000

# Definir los pesos para la capa oculta wo D capa de salida wo
wo = np.random.uniform(low=-1, high=1, size=(M, L))
wh = np.random.uniform(low=-1, high=1, size=(L, N))
