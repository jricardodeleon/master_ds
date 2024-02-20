#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:20:26 2022

@author: khrw896
"""

import numpy
import matplotlib.pyplot as plt

class LinearModel:
    def init(self):
        self.a = 0
        self.b = 0
        
    def calculateDistance(self, x, x_mean):
        return (x- x_mean)**2
    
    def train(self, dataset):
        means = numpy.mean(dataset, axis=0)
        x_mean = means[0]
        y_mean = means[1]
        
        b_numerator =  0
        b_denominator = 0
        for i in range(len(dataset)):
            b_numerator +=(dataset[i][0] - x_mean) * (dataset[i][1] - y_mean)
            b_denominator += self.calculateDistance(dataset[i][0], x_mean)
            
        self.b = b_numerator / b_denominator
        self.a = y_mean - self.b * x_mean
        
    def predict(self, x):
        return self.b * x + self.a
    
    
class RidgeModel(LinearModel):
    def init(self, lambda_):
        self.lambda_ = lambda_
        super().init()
        
    def calculateDistance(self,x,xmean):
        return super().calculateDistance(x,xmean) + self.lambda_ * self.b**2




xs= numpy.random.random(100)*100
dataset = numpy.array([[ x ,x*(1 + numpy.random.random() ) ] for x in xs])
lambda_ = 100
linear = LinearModel()
ridge = RidgeModel()
ridge.init(lambda_)

linear.train(dataset)
for i in range(3):
    print('Iteracion:', i+1)
    ridge.train(dataset)
plt.plot(dataset)
plt.show()
    #show(dataset, linear, ridge)