#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 09:16:00 2022

@author: Ricardo De Leon
"""

import numpy as np
import pandas as pd
import random

df=pd.read_excel("SA2.xlsx", sheet_name='54c_test' ,index_col=[0])

def get_kms_by_config_non_c(lst_configuration):
    total_kms=0
    for i in range(len(lst_configuration)-1):
        total_kms+=df[lst_configuration[i]] [lst_configuration[i+1]]
    return total_kms

def merge_new_combination(static, config):
    static[1:1] = config
    return static_cities

def switch_cities(lst_test_config):
    idx = range(len(lst_test_config))
    i1, i2 = random.sample(idx[1:len(idx)-1], 2)  
    lst_test_config[i1], lst_test_config[i2] = lst_test_config[i2], lst_test_config[i1]
    return lst_test_config

static_cities = list(['Copenhagen','Lynge'])
all_cities = list(df.index.values)
non_static_cities = [ele for ele in all_cities if ele not in static_cities]
init_conf=merge_new_combination(static_cities, non_static_cities)

def run_SA_algorthm(init_conf):
    t = 0;
    T = 850; # Temperatura
    k =.95
    n_config = 0
    
    while (T > .1):
        
        for _ in range(1500):
            
            n_config+=1
            curr_kms = get_kms_by_config_non_c(init_conf)    
            
            # Get a new configuration x’ close to the configuration 𝑥        
            
            test_conf=switch_cities(init_conf[:])
            new_route_kms = get_kms_by_config_non_c(test_conf)        
            # Calculate Δ𝐸 = 𝐸(𝑥′) − 𝐸(𝑥)       
            
            cost_diff = new_route_kms - curr_kms
            
            print(f'config init : kms {curr_kms} config test  kms {new_route_kms} diff {cost_diff} {T}')

            # Take 𝑞 = 𝑒−Δ𝐸⁄𝑇 and a random 0 < p < 1
            q = np.exp(-cost_diff / T)
            p = np.random.uniform(0, 1)        
            # If 𝑝 < 𝑞 make 𝑥 ← 𝑥′    
            if p < q:     
                init_conf=test_conf[:]
    
        t+=1;
        
        
        #T=T/(1 + t)
        T= k ** t * T
        
    print(f'Best Route {curr_kms} kms {get_kms_by_config_non_c(init_conf)} attempts {n_config}')
    return init_conf, get_kms_by_config_non_c(init_conf), n_config * t

sa_kms=[]
sa_route=[]
exp=3
attempts=0
for _ in range(exp):
    route, kms, attempts = run_SA_algorthm(init_conf)
    sa_route.append(route)
    sa_kms.append(kms)
    
min_route=np.where(sa_kms == np.amin(sa_kms))[0]

print('*' * 110)
print(f'route {sa_route[int(min_route)]} kms {sa_kms[int(min_route)]} n# de ciclos {attempts}')

sd=np.std(sa_kms)