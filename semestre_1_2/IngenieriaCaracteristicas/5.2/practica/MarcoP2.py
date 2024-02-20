#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:56:23 2023

@author: khrw896
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# cargar el conjunto de datos
df = pd.read_excel('practica2.xlsx')

# eliminar las observaciones sin valoración global
df = df.dropna(subset=['VALORACION GLOBAL'])

# tratar cadenas de texto en variables numéricas como datos perdidos
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' else x)

df['<5 CAUSA'] = df['<5 CAUSA'].apply(str)

# codificar la variable '<5 CAUSA' usando Feature hashing
fh = FeatureHasher(n_features=10, input_type='string')
hashed_features = fh.transform(df[['<5 CAUSA']].fillna('').astype(str).values)
hash_array = hashed_features.fit_transform(df['<5 CAUSA'].astype(str)).toarray()
# df_hash = pd.DataFrame(hash_array, columns=['fh'+str(i) for i in range(hash_array.shape[1])])
# df = pd.concat([df, df_hash], axis=1).drop('<5 CAUSA', axis=1)

# transformar las variables con una transformación de potencia para compensar el sesgo
pt = PowerTransformer(method='yeo-johnson')
df[['ADECUACIÓN COLECCIÓN', 'TRATO', 'INFORMACIÓN.1', 'VALORACION GLOBAL']] = pt.fit_transform(df[['ADECUACIÓN COLECCIÓN', 'TRATO', 'INFORMACIÓN.1', 'VALORACION GLOBAL']])

# imputar los valores faltantes por mediana
imp = SimpleImputer(strategy='median')
df[df.columns] = imp.fit_transform(df)

# modelo de árbol de decisión para predecir la valoración global
X = df.drop(['VALORACION GLOBAL'], axis=1)
y = df['VALORACION GLOBAL']
dt = DecisionTreeRegressor()
dt.fit(X, y)
