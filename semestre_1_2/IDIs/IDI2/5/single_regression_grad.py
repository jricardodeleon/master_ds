#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:35:45 2023

@author: Ricardo De Leon
"""

import sympy as sp
import numpy as np
from sigfig import round
import matplotlib.pyplot as plt


# Define sample data
np.random.seed(123)
xi = np.array([i for i in range(1,101)])
yi = np.array([xi[i] + 3 * np.random.uniform(0,1) for i in range(len(xi))])


# Define the learning rate
alpha = 0.0001

# Define the number of iterations
num_iterations = 1000

# Define the initial values for the a and b
a = 1
b = 1
iter = 1
n_error_t = 0

for i in range(num_iterations):
    iter+=i
    y_pred = a * xi + b
    error_recta = np.mean(np.sum(y_pred - yi)**2)

    grad_a = np.mean((a * xi + b - yi) * xi)
    grad_b = np.mean(a * xi + b - yi)
    
    
    a-= alpha * grad_a
    b-= alpha * grad_b       
    
    # Update variable values
    grad_lst = [grad_a,grad_b]

    # Evaluate norm to break the loop
    v = sp.Matrix(grad_lst)
    n_error = sp.trigsimp(v.norm())
    
    if(float("{:.3e}".format(n_error)) == float("{:.3e}".format(n_error_t))):
        break

    n_error_t = n_error

print(f'iter # {iter} Y = {round(a, sigfigs=4)}x + {round(b, sigfigs=4)}b, error recta = {round(error_recta, sigfigs=4)}, norma_error {round(n_error, sigfigs=4)}')
       
# Plot the dataset
plt.scatter(xi, yi)

# Calculate the final predictions for the line of regression
y_pred = a * xi + b

# Plot the line of regression
plt.plot(xi, y_pred, color='red')

# Show the plot
plt.show()    