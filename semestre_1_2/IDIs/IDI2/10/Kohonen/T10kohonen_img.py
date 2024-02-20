#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:48:28 2023

@author: khrw896
"""

from PIL import Image as im
import numpy as np

# datos de imagen
pic = im.open('imagen3Gandhi.jpg')
pic_array = np.array(pic)
altura,ancho,largo = pic_array.shape
data = np.reshape(pic_array,(altura*ancho,largo))

pic_num = "3.-"
pic_name = "imagen3Gandhi_"
pic_k = "K10_"
pic_algo = "kohonen_"
pic_ext_name = ".jpg"

X = data[:,0]
Y = data[:,1]
Z = data[:,2]

data_max = np.max(data)
data_min = np.min(data)

centros = []
centros_tmp = []


def clasifica(x,y,z):
   return np.argmin(get_distancias(x, y, z, centros),axis=0)

def get_distancias(X,Y,Z,c):
    distancias = []
    for i in range(len(c)):
        distancias.append((X - c[i][0]) ** 2 + (Y - c[i][1]) ** 2 + (Z - c[i][2]) ** 2)
    return np.array(distancias).T

def get_centros():
    cx = np.random.randint(data_min, data_max)
    cy = np.random.randint(data_min, data_max)
    cz = np.random.randint(data_min, data_max)
    return [cx,cy,cz]

def crear_centros():
    k = int(input("Cuantos centros quieres crear (Minimo 2)? "))
    for _ in range(k):
        centros.append(get_centros())

# Inicializando centros
crear_centros()

print('Centros iniciales: ')
for i in range(len(centros)):
    print(f'C{i} {centros[i]}')

# definicion de parametros de inicio

E = 1e-1
error = E + 1
centros_tmp = centros.copy()
lerning_rates = 1000


for t in range(1,lerning_rates):
        # Carrusel
        for i in range(len(X)):
            # Clasifica
            c = clasifica(X[i], Y[i], Z[i])
            # Recorre centroide
            centros[c] = 1/t  * np.array([X[i], Y[i], Z[i]]) + (1-1/t) * np.array(centros[c])
        
        error = np.max(abs(np.vstack(centros_tmp) - np.vstack(centros)))
        # print(error)
        
        # Condicion de paro
        if error < E:
            print(f'Centros Finales encontrados en la iteracion {t-1}: ')
            for i in range(len(centros)):
                print(f'C{i} {centros[i]}')
            break
        centros_tmp = centros.copy()
        
print('Error', error)

# recreando imagen
pic_mod=[]
for i in range(data.shape[0]):
    distancia =((data[i] - centros)**2).sum(axis=1) 
    pic_mod.append(centros[np.argmin(distancia)])
pic_mod= np.reshape(pic_mod,(altura,ancho,largo))

pic_output=im.fromarray(np.uint8(pic_mod))
pic_output.save(f'{pic_num}{pic_name}{pic_k}{pic_algo}{pic_ext_name}')