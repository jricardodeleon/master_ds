#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:08:38 2022

@author: khrw896
"""

class Node():
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
    def minimax(self, is_max):    
        # base case, if no sub nodes, just return the value
        if (self.left is None and self.right is None):
            return "end", self.value

        # if node has only one child
        if (self.right is None):
            l_choice = self.left.minimax(not is_max)
            return "left", l_choice
        elif (self.left is None):
            r_choice = self.right.minimax(not is_max)
            return "right", r_choice

        # if child nodes exist, run minimax on each child nodes
        l_choice = self.left.minimax(not is_max)
        r_choice = self.right.minimax(not is_max)

        # compare results
        if (is_max):
            if (l_choice > r_choice):
                return "left", l_choice
            else:
                return "right", r_choice
        else:
            if (l_choice < r_choice):
                return "left", l_choice
            else:
                return "right", r_choice




raiz=Node(0)

raiz.left=Node(4)
raiz.right=Node(9)

raiz.left.left=Node(5)
raiz.left.right=Node(2)

raiz.right.left=Node(1)
raiz.right.right=Node(-3)

raiz.left.left.left=Node(7)
raiz.left.left.right=Node(3)

raiz.left.right.left=Node(2)
raiz.left.right.right=Node(1)

raiz.right.left.left=Node(10)
raiz.right.left.right=Node(2)

raiz.right.right.left=Node(1)
raiz.right.right.right=Node(2)

res=raiz.minimax(False)