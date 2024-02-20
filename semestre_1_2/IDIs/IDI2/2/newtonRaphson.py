#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:16:49 2023

@author: Ricardo De Leon
"""

import sympy as sp
from math import pi
from sigfig import round

tolerancia = 0.00000001

def newton_raphson(x0):
    x = sp.symbols('x')
    f = input('Enter the function with variable x: ')
    df = sp.diff(f)
    f = sp.lambdify(x, f)
    df = sp.lambdify(x, df)
    tramo = abs(2 * tolerancia)
    iter_count = 1;
    while not(tramo <= tolerancia):
        x1=x0-f(x0)/df(x0)
        tramo = abs(x1-x0)
        x0 = x1
        print('x', iter_count, '=', round(x1, sigfigs=2))
        iter_count += 1

newton_raphson(pi)