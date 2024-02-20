from PIL import Image
import numpy as np
import glob
import os

def getPixeles(imagePath):
    imagen = Image.open(imagePath)
    ancho, altura = imagen.size
    array_imagen=np.array(imagen)
    return np.reshape(array_imagen,(ancho*altura,array_imagen.shape[2])), array_imagen.shape

def getFiles(path, ext):
    return glob.glob(path + "/" + ext)

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
ext='*.jpg'

imagenes=getFiles(current_dir, ext)

#Kohonen
t, error = 2,.1
k = int(input("Introduce número de centroides (Mayor o igual a 2 y Menor o igual a 32):"))
print(''.center(80,'='))
print(('Kohonen para k='+ str(k)).center(80))
print(''.center(80,'='))
for imagen in imagenes:
    data, dimensiones=getPixeles(imagen)
    image_name=imagen.split('\\')[-1]
    centroides = np.random.uniform(data.min(axis=0),data.max(axis=0),(k, data.shape[1]))
    print('Imagen:',image_name)
    while True:
        alpha=1/t
        distancia = np.zeros((k, data.shape[0]))
        for j in range(data.shape[0]):
            for i in range(k):
                distancia[i,j] =((data[j] - centroides[i])**2).sum() 
            clasificacion = np.argmin(distancia[:,j], axis=0)
            centroides[clasificacion] = alpha*data[j] + (1-alpha)*centroides[clasificacion]
        E=np.linalg.norm(alpha*(data[j]-centroides[clasificacion]))
        if E<error:
            print('Centroides (encontrados en la iteración', t-1, 'con error=',E, '):')
            break
        t+=1
    print(centroides)
    print(''.center(80,'='))

    #Clasificación
    new_image=np.empty(data.shape)
    for j in range(data.shape[0]):
        distancia =((data[j] - centroides)**2).sum(axis=1) 
        clasificacion = centroides[np.argmin(distancia)]
        new_image[j]=clasificacion
    new_image= np.reshape(new_image,dimensiones)
    #Generar nueva imagen
    ni=Image.fromarray(np.uint8(new_image))
    name=image_name.replace('.jpg','_k' + str(k)+'.jpg') 
    ni.save(name)

