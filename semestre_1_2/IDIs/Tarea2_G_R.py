#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:01:17 2023

@author: khrw896
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('forestfires.csv')

fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs = axs.ravel()

i = 0
for column in df.columns:
    if column != 'area':
        axs[i].scatter(df[column], df['area'])
        axs[i].set_title(f'{column} vs Area')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Area')
        i += 1

plt.tight_layout()
plt.show()

