#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:35:35 2023

@author: khrw896
"""

import numpy as np
import matplotlib.pyplot as plt

# Generar datos de entrenamiento aleatorios
X = np.random.rand(100, 3)
y = np.random.rand(100, 1)

# Definir los parámetros del modelo
input_size = X.shape[1]
hidden_size = 5
output_size = y.shape[1]
learning_rate = 0.1
epochs = 1000

# Inicializar los pesos y sesgos aleatoriamente
weights_ih = np.random.rand(hidden_size, input_size) - 0.5
bias_h = np.random.rand(hidden_size, 1) - 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(X, weights_ih, bias_h):
    hidden = sigmoid(weights_ih @ X.T + bias_h)
    return hidden

def backward(X, y, hidden, weights_ih, bias_h, learning_rate):
    error_h = (y.T - hidden) * sigmoid_derivative(hidden)
    weights_ih += learning_rate * error_h @ X
    bias_h += learning_rate * np.sum(error_h, axis=1, keepdims=True)
    return weights_ih, bias_h

def train_mlp(X, y, hidden_size, learning_rate, epochs):
    input_size = X.shape[1]
    weights_ih = np.random.rand(hidden_size, input_size) - 0.5
    bias_h = np.random.rand(hidden_size, 1) - 0.5
    losses = []
    for i in range(epochs):
        hidden = forward(X, weights_ih, bias_h)
        weights_ih, bias_h = backward(X, y, hidden, weights_ih, bias_h, learning_rate)
        loss = np.mean((y - hidden.T)**2)
        losses.append(loss)
    return weights_ih, bias_h, losses

# Entrenar el modelo
weights_ih_trained, bias_h_trained, losses = train_mlp(X, y, hidden_size, learning_rate, epochs)

# Imprimir los pesos y sesgos entrenados
print("Pesos de la capa oculta entrenados:\n", weights_ih_trained)
print("Sesgos de la capa oculta entrenados:\n", bias_h_trained)

# Graficar la pérdida durante el entrenamiento
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()