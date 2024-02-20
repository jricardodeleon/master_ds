#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:03:51 2022

@author: macbookpro
"""

class nodo:
    def __init__(self,info):
        self.dato = info
        self.izq = None
        self.der = None
        
        
    def minimax(self, tipo, prof):
        
        # caso base, si no hay hijos regresa la raiz y revisa profundidad
        if prof == 1:
            return self, "Hoja"
        
        if self.izq is None:
                if self.der is None:
                    return self,'hoja'
                return self.der.minimax(not tipo,prof-1)
            
        # si hay hijos, corre minimax para cada lado y para cada hijo
        nizq,_ = self.izq.minimax(not tipo, prof -1)
        nder,_ = self.der.minimax(not tipo, prof -1)
        # comparar resultados
        if tipo:
            if nizq.dato<nder.dato:
                return nder,'derecha'
            return nizq,'izquierda'
             
        if nizq.dato>nder.dato:
            return nder,'derecha'
        return nizq,'izquierda'
     
    
        

raiz=nodo(0)

raiz.izq=nodo(4)
raiz.der=nodo(9)

raiz.izq.izq=nodo(5)
raiz.izq.der=nodo(2)

raiz.der.izq=nodo(1)
raiz.der.der=nodo(-3)

raiz.izq.izq.izq=nodo(7)
raiz.izq.izq.der=nodo(3)

raiz.izq.der.izq=nodo(2)
raiz.izq.der.der=nodo(1)

raiz.der.izq.izq=nodo(10)
raiz.der.izq.der=nodo(4)

raiz.der.der.izq=nodo(1)
raiz.der.der.der=nodo(8)


res_p=raiz.minimax(False, 3)

print(res_p[0].dato)