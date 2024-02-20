#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:13:57 2022

@author: Ricardo De Leon
"""

import numpy as np
from sys import exit


class nodo:
    def __init__(self,info):
        self.dato = info
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        
        
    def minimax(self, tipo, prof):
        
        # caso base, si no hay hijos regresa la raiz y revisa profundidad
        if prof == 1:
            return self, "Hoja"
        
        if self.left is None:
                if self.right is None:
                    return self,'hoja'
                return self.right.minimax(not tipo,prof-1)
            
        if self.up is None:
                if self.down is None:
                    return self,'hoja'
                return self.down.minimax(not tipo,prof-1)
            
        # si hay hijos, corre minimax para cada lado y para cada hijo
        
        nleft,_ = self.left.minimax(not tipo, prof -1)
        nright,_ = self.right.minimax(not tipo, prof -1)
        nup,_ = self.up.minimax(not tipo, prof -1)
        ndown,_ = self.down.minimax(not tipo, prof -1)
        
        # comparar resultados
        
        if tipo:
            if nleft.dato<nright.dato:
                return nright,'right'
            return nleft,'left'
             
            if nleft.dato>nright.dato:
                return nright,'right'
            return nleft,'left'
        
            if nup.dato<ndown.dato:
                return nup,'up'
            return ndown,'down'
             
            if nup.dato>ndown.dato:
                return nup,'up'
            return ndown,'down'
    
    
    
    

class Board_game():
    
    def __init__(self, size):
        self.size = size
        self.p1_x = 0
        self.p1_y = 0
        self.p2_x = size - 1
        self.p2_y = size - 1
        self.board = self.init_board(size)
        self.curr_score_p1 = self.board[self.p1_x,self.p1_y]
        self.curr_score_p2 = self.board[self.p2_x, self.p2_y]
        self.total_match = 0
        self.can_move = True
    
    def init_board(self, size):
        np.random.seed(12)
        return np.random.randint(1,10, size = (size, size))
    
    def print_board(self):
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.board]))
        
    def user_move(self):
        direction = input('Enter "R" to move to right "L" for left "U" for up "D" for Down or "X" to exit: ')
        
        while True:
            
            if(direction.lower() not in ['u', 'd', 'l', 'r', 'x']):
                return print('Enter a valid Option \n')

            if direction == "x" :
                exit()
                
            if direction == "u" and self.p2_x >= 0:
                if self.board[self.p2_x - 1, self.p2_y] != 11:
                    self.board[self.p2_x, self.p2_y] = 11
                    self.p2_x -= 1
                    self.curr_score_p2 += self.board[self.p2_x, self.p2_y]
                    print(f'Human new position at coordinates {self.p2_x} {self.p2_y} with value {self.board[self.p2_x, self.p2_y]}')
                    break
                else: 
                    print('position not allowed try again')
                    self.user_move()
                    break
            
            if direction == "d" and self.p2_x >= 0:
                if self.board[self.p2_x + 1, self.p2_y] != 11:
                    self.board[self.p2_x, self.p2_y] = 11
                    self.p2_x += 1
                    self.curr_score_p2 += self.board[self.p2_x, self.p2_y]
                    print(f'Human new position at coordinates {self.p2_x} {self.p2_y} with value {self.board[self.p2_x, self.p2_y]}')
                    break
                else: 
                    print('position not allowed try again')
                    self.user_move()
                    break
            
            if direction == "l" and self.p2_y >= 0:
                if self.board[self.p2_x, self.p2_y - 1] != 11:
                    self.board[self.p2_x, self.p2_y] = 11
                    self.p2_y -= 1
                    self.curr_score_p2 += self.board[self.p2_x, self.p2_y]
                    print(f'Human new position at coordinates {self.p2_x} {self.p2_y} with value {self.board[self.p2_x, self.p2_y]}')
                    break
                else: 
                    print('position not allowed try again')
                    self.user_move()
                    break
            
            if direction == "r" and self.p2_y >= 0:
                if self.board[self.p2_x, self.p2_y + 1] != 11:
                    self.board[self.p2_x, self.p2_y] = 11
                    self.p2_y += 1
                    self.curr_score_p2 += self.board[self.p2_x, self.p2_y]
                    print(f'Human new position at coordinates {self.p2_x} {self.p2_y} with value {self.board[self.p2_x, self.p2_y]}')
                    break
                else: 
                    print('position not allowed try again')
                    self.user_move()
                    break
                
    def check_if_human_can_move(self):
        if self.board[self.p2_x - 1, self.p2_y] == 11 and self.board[self.p2_x + 1, self.p2_y] == 11 \
        and self.board[self.p2_x, self.p2_y - 1] == 11 and self.board[self.p2_x, self.p2_y + 1] == 11:
            self.can_move = False

    def board_match(self):
        self.total_match =+ self.curr_score_p1 - self.curr_score_p2
        return self.total_match
        
    def play_game(self):
        self.print_board()
        print('Initial match')
        print(f'Score {self.board_match()}')
        if self.board_match() > 0:
            print('AI winning')
        else:
            print('Human winning')
        while self.can_move:
                print(f'Score for AI player {self.curr_score_p1} Score for human player {self.curr_score_p2}')
                self.user_move()
                self.print_board()
                self.check_if_human_can_move()
                
                if not self.can_move:
                    print("You can't make more movements")
                    if self.board_match() > 0:
                        print('AI won')
                    else:
                        print('Human won')
                
                #break

m=Board_game(5)
m.play_game()

