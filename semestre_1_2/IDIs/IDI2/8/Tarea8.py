#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:34:35 2023

@author: Riardo de Leon
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
x = np.random.randint(1,100,50)
y = np.random.randint(1,100,50)

plt.scatter(x, y, c = "blue")

data = np.asarray([x,y]).T

data_max = np.max(data)
data_min = np.min(data)

def get_centros():
    cx = np.random.randint(data_min, data_max)
    cy = np.random.randint(data_min, data_max)
    return np.asarray([cx,cy])

centro_1 = get_centros()
centro_2 = get_centros()