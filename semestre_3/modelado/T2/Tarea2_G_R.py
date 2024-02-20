#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:01:17 2023

@author: khrw896
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('forestfires.csv')


columns = list(df.columns.values)

##Plot test input distriubtions
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 5))
axs = axs.ravel()


for i in range(len(columns) - 1):

    axs[i].scatter(df[columns[i]], df['area'])
    axs[i].set_title(f'{columns[i]} vs area')
    axs[i].set_xlabel(columns[i])
    axs[i].set_ylabel('Area')
    
plt.tight_layout()
plt.show()

