

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

data = pd.read_excel('practica2.xlsx')
data.head()

"""Convertir datos perdidos a NaN y tambien aquellos que no son numericos se cambiaran por NaN"""

data["VALORACION GLOBAL"] = pd.to_numeric(data["VALORACION GLOBAL"], errors="coerce")

"""Considere las cadenas de texto en variables numéricas como datos perdidos."""

columnas_numericas = data.columns[:10].tolist()
for col in columnas_numericas:
    data[col] = pd.to_numeric(data[col], errors='coerce')

"""Eliminar filas con NaN en la columna "VALORACION GLOBAL"
"""

data = data.dropna(subset=["VALORACION GLOBAL"])
data = data.reset_index(drop = True)

"""Reemplazar los NaN por "DELETE_WORD"
"""

data["<5 CAUSA"].fillna("DELETE_WORD", inplace=True)
data['<5 CAUSA'] = data['<5 CAUSA'].apply(str) #"Hay valores numericos y marca error"

"""Analisis TF-IDF """

tfidf_vectorizer = TfidfVectorizer(max_df=1.0)
tfidf = tfidf_vectorizer.fit_transform(data["<5 CAUSA"])


#  Obtener la matriz de características
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_values = tfidf.toarray()
tfidf_df = pd.DataFrame(tfidf_values, columns=feature_names)
tfidf_df  = tfidf_df.drop('delete_word', axis=1)

# Calculamos la norma TF IDF
normas = np.linalg.norm(tfidf_df, axis=0)

import matplotlib.pyplot as plt

x_sorted = np.sort(normas)[::-1] # ordenar de mayor a menor
plt.plot(x_sorted)
plt.show()

print(x_sorted[:20])

"""Tomarelos las primeras 8 caracteristicas o palabras, para el feature hashing"""

hashing_vectorizer = HashingVectorizer(n_features=8)
hashed_matrix =   hashing_vectorizer.transform(data["<5 CAUSA"])

# convertir la matriz de características en un DataFrame
hashed_df = pd.DataFrame(hashed_matrix.toarray())

# eliminar la columna "<5 CAUSA>" original del dataset
data.drop('<5 CAUSA', axis=1, inplace=True)

missing_values_percentage = data.isnull().mean() * 100
print(missing_values_percentage)

"""Transformacion de potencia """

print(data.skew())
data.hist(bins=50)

columnas_a_transformar = data.iloc[:, :10]
# crear una instancia de la transformación de potencia
transformer = PowerTransformer(method='yeo-johnson')
columnas_transformadas = transformer.fit_transform(columnas_a_transformar)

# asignar las columnas transformadas al DataFrame original
data.iloc[:, :10] = columnas_transformadas

print(data.skew())
data.hist(bins=50)

# completar las columnas que tienen datos perdidos, se usara la mediana
for col in columnas_numericas:
  moda = data[col].median()
  data[col].fillna(moda, inplace=True)

# # Eliminar columnas con gran cantidad de valores faltantes
# data = data.drop(['WIFI.1', 'INFORMACIÓN.1'], axis=1)

# # Renombramos las columnas agregando el prefijo "Word-"
# hashed_df = hashed_df.add_prefix('Word-')
# # agregar las nuevas columnas a nuestro dataset
# data = pd.concat([data, hashed_df], axis=1)
# # mover al final
# columna_valor = data['VALORACION GLOBAL']
# data = data.drop('VALORACION GLOBAL', axis=1)
# data['VALORACION GLOBAL'] = columna_valor

# data

# # Aplicar modelo de arboles de decision:
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# X = data.drop("VALORACION GLOBAL", axis=1)
# y = data["VALORACION GLOBAL"]


# # seleccionar las características y la variable objetivo
# X = data.drop("VALORACION GLOBAL", axis=1)
# y = data["VALORACION GLOBAL"]

# # dividir el conjunto de datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # crear el modelo de árbol de decisión
# tree_model = DecisionTreeRegressor(random_state=1)

# # entrenar el modelo utilizando el conjunto de datos de entrenamiento
# tree_model.fit(X_train, y_train)

# y_pred = tree_model.predict(X_test)
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print(f"Error cuadrático medio: {mse}")
# print(f"Raíz del error cuadrático medio: {rmse}")


# # evaluar el modelo utilizando el conjunto de datos de prueba
# score = tree_model.score(X_test, y_test)
# print("El rendimiento del modelo de árbol de decisión es:", score)

# # obtener la importancia de las características
# feature_importances = tree_model.feature_importances_

# # imprimir la importancia de cada característica
# for i, feature_name in enumerate(X.columns):
#     print("La importancia de la característica", feature_name, "es:", feature_importances[i])