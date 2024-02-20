#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:48:53 2023

@author: khrw896
"""


import sympy as sp
import numpy as np
import pandas as pd

df = pd.read_excel('/Users/khrw896/Documents/Master/IDI/IDI2/5/tareaRGD.xlsx')

X = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])

# Initialize a and b
_, n_variables = X.shape
a = np.ones(n_variables)
b = 1


# Define the learning rate
alpha = 0.0001

# Define the number of iterations
num_iterations = 100000
iter = 1
y_pred_finals = []
norm_errors_finals = []
n_error_t = 0


for i in range(num_iterations):
    grad_lst = []
    iter += 1
    
    y_pred = np.dot(X, a) + b
    error_recta = np.mean(np.sum(y_pred - y)**2)

    grad_a = -2*np.dot(X.T, y - y_pred)
    grad_b = -2*np.mean(y - y_pred)
    
    a -= alpha * grad_a
    b -= alpha * grad_b
    
    # Update variable values
    for i in range(len(grad_a)):
        grad_lst.append(grad_a[i])
        
    grad_lst.append(grad_b)

    # Evaluate norm to break the loop
    v = sp.Matrix(grad_lst)
    n_error = sp.trigsimp(v.norm())
    
    if(float("{:.4e}".format(n_error)) == float("{:.4e}".format(n_error_t))):
        break
    n_error_t = n_error
    
print(f'iter # {iter} Y = {"{:.3e}".format(a[0])}x1 + {"{:.3e}".format(a[1])}x2 + {"{:.3e}".format(a[2])}x3 + {"{:.3e}".format(b)}b,  norma_error {n_error}, error recta {"{:.3e}".format(error_recta)}')
    
