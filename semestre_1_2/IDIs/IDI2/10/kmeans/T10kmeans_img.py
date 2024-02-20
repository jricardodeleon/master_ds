# -*- coding: utf-8 -*-
"""
Spyder Editor

This Ricardo De Leon
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
pic_k = "K32_"
pic_algo = "kmeans_"
pic_ext_name = ".jpg"

X = data[:,0]
Y = data[:,1]
Z = data[:,2]

data_max = np.max(data)
data_min = np.min(data)

centros = []

def get_medias(asignaciones):
    medias = []
    for i in range(len(np.unique(asignaciones))):
        media_x = np.mean(X, where=asignaciones==i)
        media_y = np.mean(Y, where=asignaciones==i)
        media_z = np.mean(Z, where=asignaciones==i)
        if np.isnan(media_x) or np.isnan(media_y) or np.isnan(media_z):
            medias.append([X[i], Y[i], Z[i]])
        else:    
            medias.append([media_x, media_y, media_z])
    return medias
    
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
    
counter = 0

while True:
    
    counter += 1
    # Medimos distancias al centro
    dist = get_distancias(X,Y,Z, centros)
    
    # Clasificamos
    asignaciones = np.argmin(dist,axis=1)
    
    # Obtenemos Medias
    NCS = get_medias(asignaciones)
    
    # Condicion para parar si 
    if centros == NCS:
        print(f'Centros Finales encontrados en la iteracion {counter}: ')
        for i in range(len(NCS)):
            print(f'C{i} {NCS[i]}')
        break

    centros = NCS

# recreando imagen
pic_mod=[]
for i in range(data.shape[0]):
    distancia =((data[i] - centros)**2).sum(axis=1) 
    pic_mod.append(centros[np.argmin(distancia)])
pic_mod= np.reshape(pic_mod,(altura,ancho,largo))

pic_output=im.fromarray(np.uint8(pic_mod))
pic_output.save(f'{pic_num}{pic_name}{pic_k}{pic_algo}{pic_ext_name}')



