import numpy as np
import pandas as pd

df=pd.read_excel('tabla_para_probar.xlsx')
t=df.replace('?', np.nan).dropna()
r= pd.merge(df,t, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

alpha,n=0.1, 100000

inputs=[c for c in t.columns if 'x' in c]
outputs=[c for c in t.columns if 'd' in c]

N=len(inputs)
M=len(outputs)
Q=len(t.index)
L=10 #Hidden neurons

X = np.array(t[inputs], dtype=float)
D = np.array(t[outputs], dtype=float)

wo = np.random.uniform(low=-1, high=1, size=(M, L))
wh = np.random.uniform(low=-1, high=1, size=(L, N))

#entrenamiento
for i in range(n):
    dwh = np.zeros_like(wh)
    dwo = np.zeros_like(wo)
    for j in range(Q):
        #forward
        neth = wh @ X[j].T
        yh = 1 / (1 + np.exp(-neth))
        neto = wo @ yh
        y = 1 / (1 + np.exp(-neto))
        #backward
        do = (D[j].T - y) * (y * (1 - y))
        dh = yh * ( 1 - yh) * (wo.T @ do)
        #aprendizaje
        dwo += np.reshape(alpha*do,(M,1)) * np.reshape(yh.T,(1,L))
        dwh += np.reshape(alpha*dh,(L,1)) * np.reshape(X[j],(1,N))
        error=np.linalg.norm(do)
        wo = wo+dwo
        wh = wh+dwh
    if error < 0.001:
        print('El error es:', error, 'encontrado en la iteración:', i)
        break

# #reconocimiento
XR = np.array(r[inputs], dtype=float)
for j in range(len(r.index)):
    #forward
    neth = wh @ XR[j].T
    yh = 1 / (1 + np.exp(-neth))
    neto = wo @ yh
    y = 1 / (1 + np.exp(-neto))
    #Resultados
    print('Para las entradas:', XR[j], ' las salidas son:',np.round_(y,0))
    



