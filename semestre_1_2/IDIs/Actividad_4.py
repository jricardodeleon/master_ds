#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:45:40 2022

@author: Ricardo De Leon
"""

import numpy as np

# Genere una lista L con los resultados de emular 1000 veces el experimento de obtener la suma de dos dados. 
# Suponga que los dados están cargados y el 5 tiene el doble de probabilidades que los demás. 
# ¿Qué porcentaje de los resultados fue 10?

L=(np.sum(np.sum(np.random.choice([1,2,3,4,5,5,6], (1000,2)), 1) == 10) / 1000) * 100
print(f'{L}%')


# Genere una lista M con los resultados de emular las respuestas aleatorias de 1000 exámenes de 8 preguntas verdadero-falso.  
# ¿Cuál fue el promedio de calificación sobre 100?

M=np.mean((np.random.binomial(8, 0.5, 1000) * 100)/8)


# Se sabe que por un crucero pasan en promedio 20 coches por minuto. Emule aleatoriamente los resultados de tomar mediciones cada minuto durante una hora. 
# ¿Cuál fue el mayor y el menor valor obtenido?

poisson=np.random.poisson(20, 60)
print(f'Mayor Valor {np.max(poisson)}, Menor valor {np.min(poisson)}')

# El coeficiente intelectual (IQ) es un estimador de la inteligencia general. Se distribuye normalmente con una media de 100 y desviación estándar de 15.  
# Genere una lista Q que emule los IQ de una muestra aleatoria de 500 personas. Si se considera a una persona superdotada si su IQ es igual o mayor a 130, 
# ¿cuántos superdotados se obtuvieron en la simulación?

Q=np.sum(np.random.normal(100, 15, 500)>=130)


