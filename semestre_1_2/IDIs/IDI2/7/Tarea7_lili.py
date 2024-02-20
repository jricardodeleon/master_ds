import numpy as np
import pandas as pd

def distribucionDF(df,agregador,nombre):
    print(''.center(70,'='))
    dfDist= df.groupby([agregador])[agregador].count().reset_index(name='Total')
    dfDist['Percentage']=dfDist['Total']/len(df.index)
    print('Distribución set ', nombre, ':\n',dfDist)

def forward(wh, wo, xj):
    neth = wh @ xj.T
    yh = 1 / (1 + np.exp(-neth))
    neto = wo @ yh
    y = 1 / (1 + np.exp(-neto))
    return yh, y

def testing(wh, wo, X):
    R=[]
    for j in range(len(X)):
        #forward
        yh,y=forward(wh,wo,X[j])
        R.append(round(y[0]))
    return R

def accuracy(R,D,nombre):
    a=[1 for i in range(len(R)) if R[i]==D[i]]
    print(nombre,':\n\t',sum(a),' coincidencias de', len(R), '\n\tAccuracy:',sum(a)/len(R))
   

df=pd.read_excel('PercMultAplicado.xlsx')
df=df.rename(columns={'Antigüedad laboral (meses)': "Antigüedad"})
df['CargaN']=df['Mensualidad']/df['Ingreso mensual']
df['Salida']=df['Mora'].map({'SI': 1, 'NO': 0})

#Normalización de Monto y Antigüedad
df['Raiz(Monto)']=np.sqrt(df['Monto'])
df['MontoN']=(df['Raiz(Monto)'] - df['Raiz(Monto)'].mean()) / df['Raiz(Monto)'].std()
df['AntigüedadN']=(df['Antigüedad']-df['Antigüedad'].min())/(df['Antigüedad'].max()-df['Antigüedad'].min())

#Gráficas de distribuciones
df['CargaN'].plot(kind='kde',title='Distribución CargaN')
df['MontoN'].plot(kind='kde',title='Distribución MontoN')
df['AntigüedadN'].plot(kind='kde',title='Distribución Antigüedad')

#Asignar test & training 
train0=df[df['Salida']==0].sample(frac=0.7, replace=False, random_state=1)
train1=df[df['Salida']==1].sample(frac=0.7, replace=False, random_state=1)
train=pd.concat([train0,train1])
train=train[['CargaN','MontoN','AntigüedadN','Salida']]

test=df[['CargaN','MontoN','AntigüedadN','Salida']]
merged_df = pd.merge(df, train, how='left', indicator=True)
test = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

#Validar distribucion de los sets
distribucionDF(df,'Salida','original')
distribucionDF(train,'Salida','Train')
distribucionDF(test,'Salida','Test')

alpha,n=0.1, 100000

inputs=['CargaN','MontoN','AntigüedadN']
outputs=['Salida']

N=len(inputs)
M=len(outputs)
Q=len(train.index)
L=5 #Hidden neurons

X = np.array(train[inputs])
D = np.array(train[outputs])

wo = np.random.uniform(low=-1, high=1, size=(M, L))
wh = np.random.uniform(low=-1, high=1, size=(L, N))

#Training
for i in range(n):
    for j in range(Q):
        #forward
        yh,y=forward(wh,wo,X[j])
        #backward
        do = (D[j].T - y) * (y * (1 - y))
        dh = yh * ( 1 - yh) * (wo.T @ do)
        #aprendizaje
        wo+= np.reshape(alpha*do,(M,1)) * np.reshape(yh.T,(1,L))
        wh+= np.reshape(alpha*dh,(L,1)) * np.reshape(X[j],(1,N))        
        error=np.linalg.norm(do)
    if error < 0.001:
        print('Results'.center(70,'='))
        print('El error es:', error, 'encontrado en la iteración:', i + Q)
        break

#Testing
rTrain=testing(wh,wo,np.array(train[inputs]))
rTest=testing(wh,wo,np.array(test[inputs]))

print(''.center(70,'='))
#Accuracy
accuracy(rTrain,D,'Training')
T=np.array(test[outputs])
accuracy(rTest,T,'Testing')
