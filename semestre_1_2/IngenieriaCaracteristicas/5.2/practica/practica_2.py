#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:09:07 2023

@author: khrw896
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('practica2.xlsx')


df = df[df['VALORACION GLOBAL'].astype(str).str.isdigit()]
df = df.reset_index(drop=True)
df['VALORACION GLOBAL'].sort_values().unique()


print(df.skew())

from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=5, input_type='string')
hf = h.transform(df[['<5 CAUSA']].fillna('').astype(str).values)
df_h = pd.DataFrame(hf.toarray(), columns=[f'CAUSA_{i}' for i in range(5)])
df = pd.concat([df.drop('<5 CAUSA', axis=1), df_h], axis=1)
