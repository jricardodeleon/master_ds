#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:30:57 2023

@author: khrw896
"""

import pandas as pd

# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# Configuración warnings
# ==============================================================================
import warnings
# warnings.filterwarnings('ignore')

#%%

# conjuntos de datos de los viajes realizados en la ciuad de mexico en colectivo
df = pd.read_csv('c_de_datos_mexico.csv')

# unificación de columnas ANIO y ID_MES a FECHA
df['FECHA'] = pd.to_datetime(df.ANIO.astype(str) + '/' + df.ID_MES.astype(str) + '/01')

# hacer la columna de fecha como index
#df = df.set_index('FECHA')

df = df.rename(columns={'VALOR': 'y'})
# df = df.asfreq('MS')
df['FECHA'] = df['FECHA'].sort_index()
df.head()


print(f'Número de filas con missing values: {df.isnull().any(axis=1).mean()}')

#%%
(df.index == pd.date_range(
                    start = df.index.min(),
                    end   = df.index.max(),
                    freq  = df.index.freq)
).all()
#%%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df['FECHA'], df['y']);
#%%