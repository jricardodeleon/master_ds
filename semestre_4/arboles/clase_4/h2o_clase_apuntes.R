# install.packages("/Users/macbookpro/Downloads/h2o_3.44.0.3.tgz",
#                  repos = NULL, type = "source")
# install.packages("h2o")
library(h2o)
library(tidyverse)


#Inicializacion h2o

h2o.init(ip="localhost",
         #-1 indica que utilicen todos los cores disponible
         nthreads = -1,
         #Maxima memoria disponible
         max_mem_size= "8g")

h2o.removeAll()

h2o.no_progress()
# install.packages("mosaicData")
library(mosaicData)

data("SaratogaHouses")
data <- SaratogaHouses

datos_h2o <- as.h2o(x=data,destination_frame = "datos_h2o")

#Dimensiones frame

h2o.dim(datos_h2o)

#Nombre de las columnas

h2o.colnames(datos_h2o)

h2o.describe(datos_h2o)


datos_h2o[1]
datos_h2o$price

# ver la  correlacion de los datos para las columnas numericas

h2o.columns_by_type(object=datos_h2o,coltype = "numeric")

indices <- datos_h2o %>% h2o.columns_by_type(coltype = "numeric")

h2o.colnames(x=datos_h2o)[indices]

h2o.cor(x=datos_h2o[indices],method = "Pearson")

as.data.frame(h2o.table(datos_h2o$fuel)) %>% 
  ggplot(aes(x=fuel,y=Count,fill=fuel)) +
  geom_col()

#Split de los datos

#Train 60% 
#Validacion 20%
#Test 20%

h2o.dim(datos_h2o)

split_h2o <- h2o.splitFrame(data=datos_h2o,ratios = c(0.6,.2),seed = 123)

train_h2o <- h2o.assign(data = split_h2o[[1]],key="train_h2o") #Train 60%
h2o.dim(train_h2o)

validation_h2o <- h2o.assign(data = split_h2o[[2]],key="validation_h2o") #Validacion 20%
h2o.dim(validation_h2o)

test_h2o <- h2o.assign(data = split_h2o[[3]],key="test_h2o") #Test 20%
h2o.dim(test_h2o)

# Random Forest
# Optimizacion

# Definimos el grid

hypergrid_RF <- list(ntrees = seq(50,500, by=50), # Numero de aarboles
                     mtries = seq(4,5, by=1) # Number de variables que se eligen en cada arbol
)

# optimizar

grid_RF <- h2o.grid(
  # definimos el algoritmo
  algorithm = "randomForest",
  # variable de respuesta
  y = 1,
  # predictores
  x = c(2:16),
  # datos de entrenamiento
  training_frame = train_h2o,
  # datos de validacion
  validation_frame = validation_h2o,
  # preprocesado (varianza cero) quitar todas las variavle que no tengan cambios o que sean constates
  ignore_const_cols=TRUE,
  # merica optimizacion
  stopping_metric = "RMSE",
  # hiperparametros
  hyper_params = hypergrid_RF,
  # tipo de busqueda
  search_criteria = list(strategy="Cartesian"),
  seed=123,
  grid_id = "grid_RF"
)

# resumen del grid
res_grid_RF <- h2o.getGrid(grid_id = "grid_RF", 
                           sort_by = "RMSE", 
                           decreasing = FALSE)

# guardar el mejor modelo basado en la metrica rmse del train
best_model <- h2o.getModel(res_grid_RF@model_ids[[1]]) # el mejor modelo es el 1 por que asi lo pusimos a la hora de hacer el resumen decreasing
best_model

# prediccion
pred_RF <- h2o.predict(best_model, test_h2o)

# error del test
rmse_test <- h2o.performance(model = best_model, newdata = test_h2o)@metrics$RMSE


train_h2o[1]
