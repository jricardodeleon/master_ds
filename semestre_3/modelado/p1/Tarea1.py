import pandas as pd
import numpy as np
import sklearn.metrics as skm # similarity metrics
import scipy.spatial.distance as sc # distance metrics
from scipy.spatial.distance import hamming

#%% Import data
file_path = "Predictive_Mod_Oto_2023.xlsx"
data = pd.read_excel(file_path,index_col=0)
data.head()

#%% Select columns
def select_columns(x):
  csel = np.arange(10,70,3)
  users1 = list(x.iloc[:,7])
  cnames1 = list(x.columns.values[csel])
  x = x[cnames1]
  x.index = users1
  return x

datan =  select_columns(data)
datan.head()

#%% Validate if there are Nan in the data
nan_check = datan.isna().any().any()

if nan_check:
    print("There are NaN values in the file")
else:
    print("There are not NaN values in the file")

#%% Change Yes to 1 and No to 0
print('Execution types:\n\t1) Convert the "yes" answers to 1 and "no" to 0\n\t2) Convert the "yes" answers to 0 and "no" to 1\n')
tipo=int(input('Please enter execution type (1 o 2):'))
cnames = list(datan.columns.values)
fnames = np.array(datan.index)
for col in cnames:
    if tipo==1:
        datan[col]=np.where(datan[col]=="Yes",1,0)
    else:
        datan[col]=np.where(datan[col]=="No",1,0)
datan.head()
    
#%% Calculate similarity indices in users by sklearn
print('Calculation of distances by sklearn:')
cf_m = skm.confusion_matrix(datan.iloc[0,:],datan.iloc[1,:])
sim_simple = skm.accuracy_score(datan.iloc[0,:],datan.iloc[1,:])
#sim_simple_new = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m)
print('\tSimple : %0.4f'%sim_simple)
sim_jac = skm.jaccard_score(datan.iloc[0,:],datan.iloc[1,:])
#sim_jac = (cf_m[1,1])/(np.sum(cf_m)-cf_m[0,0])
print('\tJaccard: %0.4f'%sim_jac)


#%% Calculation of distances by scipy
print('Calculation of distances by scipy:')
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
tt = datan.iloc[1,:]
d1 = sc.hamming(datan.iloc[0,:],datan.iloc[1,:])
print('\tSimple : %0.4f'%d1)
d2 = sc.jaccard(datan.iloc[0,:],datan.iloc[1,:])
print('\tJaccard: %0.4f'%d2)

#%% Calculate all possible combinations by scipy
D1 = sc.pdist(datan,'matching')
D1 = sc.squareform(D1)

D2 = sc.pdist(datan,'jaccard')
D2 = sc.squareform(D2)

#%% Select a user and determine the other most similar user
#Ask for an user and get the most similar
user_name= input("\nPlease enter your alias: ")
user = data[data['Alias'] == user_name].index.item() -1 

#Matching
DM_user = D1[user]
DM_user_sort = np.sort(DM_user)
indx_user_m = np.argsort(DM_user)

#Jaccard
DJ_user = D2[user]
DJ_user_sort = np.sort(DJ_user)
indx_user_j = np.argsort(DJ_user)

#%% The most similar user
User = datan.loc[fnames[user]]

User_sim_m = datan.loc[fnames[indx_user_m[1]]]
print('\nMost similar user (Matching):')
print('\t',User_sim_m.name)

User_sim_j = datan.loc[fnames[indx_user_j[1]]]
print('\nMost similar user (Jaccard):')
print('\t',User_sim_j.name)