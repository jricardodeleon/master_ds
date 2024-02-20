import numpy as np
import pandas as pd
import random 

df=pd.read_excel("SA2.xlsx", index_col=[0])

def get_kms_by_config_non_c(lst_configuration):
    total_kms=0
    for i in range(len(lst_configuration)-1):
        total_kms+=df[lst_configuration[i]] [lst_configuration[i+1]]
    return total_kms

def switch_cities(config):
    idx = range(len(config))
    #i1, i2 = random.sample(idx, 2)       ######## PARA RUTAS NO CICLICAS ########
    i1, i2 = random.sample(idx[1:len(idx)-1], 2)      ######## PARA RUTAS NO CICLICAS E INICIO & FINAL FIJO ########
    config[i1], config[i2] = config[i2], config[i1]
    return config

# Initial Config
#init_config=list(df.index.values)
######## PARA RUTAS NO CICLICAS ########
index_config=list(df.index.values) 
init_conf=list([index_config[6]]+index_config[:6] + index_config[6+1:33]+index_config[33+1:54] +[index_config[33]])

runs=10
results=np.empty((0,4), int)
n_config = 0

for i in range(runs):
    # Parameters
    t=0
    T=350
    r=1000
    k=0.95

    # Annealing
    while (T > 1):    
        for _ in range(r):
            n_config+=1
            curr_kms = get_kms_by_config_non_c(init_conf)
            # Get a new configuration x’ close to the configuration 𝑥
            test_conf=switch_cities(init_conf)
            new_route_kms = get_kms_by_config_non_c(test_conf)
            # Calculate Δ𝐸 = 𝐸(𝑥′) − 𝐸(𝑥)
            cost_diff = new_route_kms - curr_kms
            print(f'config init : kms {curr_kms} config test  kms {new_route_kms} diff {cost_diff} {T}')
            # Take 𝑞 = 𝑒−Δ𝐸⁄𝑇
            q = np.exp(- round(cost_diff,2) / T)
            p = np.random.uniform(0, 1)
            # If 𝑝 < 𝑞 make 𝑥 ← 𝑥′
            if p < q:
                init_conf=test_conf[:]
        t+=1
        #T=T/(1 + t)
        T=0.95*T
    print(f'Mejor ruta encontrada {i+1} de {runs}: {init_conf}.')
    print(f'{get_kms_by_config_non_c(init_conf)}kms.')
    print(f'{t*r} intentos')
    