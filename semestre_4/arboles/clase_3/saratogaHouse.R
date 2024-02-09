#install.packages("mosaicData")
#install.packages("tidymodels)

library(mosaicData)
library(tidyverse)
library(tidymodels)

data("SaratogaHouses")
data <- SaratogaHouses


colnames(data)

summary(data)

# split train / test

set.seed(123)

split_initial <- initial_split(data = data,
              prop = 0.8, # training proportion
              strata = price, # Variable objetivo
              # hace el split lo más distribuido del precio automaticamente
              )

data_train <- training(split_initial)
data_test <- testing(split_initial)

summary(data_train)
summary(data_test)

# receta/recipies

# Definninr las transformaciones
# recipie()

transformer <- recipe(formula = price ~., # el precio es mi objetivo ~. esto signidica que todas las demás son variables predictoras
        data = data_train) |>
        # step_mutate(value_per_acres = landValue / lotSize) |> # agregar columna nueva de valor de metros cuadrados
        step_naomit(all_predictors()) |> # para omitir los valores nulos
        step_nzv(all_predictors()) |> # varianza a todos los predictores
        step_center(all_numeric(), -all_outcomes()) |> # centrar los datos
        step_scale(all_numeric(), -all_outcomes()) |> # escalar los datos
        step_dummy(all_nominal(), -all_outcomes()) # transformar en dummies las que sean nominales
        
transformer    

# Training recipe
# prep()

transformer_fit <- prep(transformer)

# aplicar las transformaciones a los dataser
# bake()

data_train_prep <- bake(transformer_fit, new_data = data_train)
data_test_prep <- bake(transformer_fit, new_data = data_test)
        
        
# Modelo
# Elegir el modelo y definir los hiperparametros a optimizar

model_tree <- decision_tree(mode = "regression", # regression, classification
             tree_depth = tune(), # tuneo de variables
             min_n = tune()) |> 
  set_engine(engine = "rpart") # paqueteria

# cross validation
cv_folds <- vfold_cv(data = data_train, # datos sin preprocesar
         v = 5, # numero de particiones
         strata = price # distribuirla de manera equitativa
         )        

# grid aleatorio
hiperpar_grid_1 <- grid_random(
  # rango de busqueda de los hiperparametros
  tree_depth(range = c(1,10), trans = NULL),
  min_n(range = c(2,50), trans = NULL),
  # puede hacer 490 combinaciones 10 * 49  que viene de (2,50) combinaciones
  # pero no vamos a poner el maximo asi que ponemos 100
  size = 100
)

# grid search
hiperpar_grid_2 <- grid_regular(
  # rango de busqueda de los hiperparametros
  tree_depth(range = c(1,10), trans = NULL),
  min_n(range = c(2,50), trans = NULL),
  # definir numero de valores para cada hiperparametro
  levels = c(10,49)
)

# grid search
hiperpar_grid_3 <- grid_regular(
  # rango de busqueda de los hiperparametros
  tree_depth(range = c(1,10), trans = NULL),
  min_n(range = c(2,50), trans = NULL),
  # definir numero de valores para cada hiperparametro
  levels = c(10,10)
)
        

# optimización de los hiperparametros

grid_fit <- tune_grid(
  object = model_tree,
  # preprocesamiento de los datos (recipe)
  preprocessor = transformer,
  # resample que se crearon sin procesar
  resamples = cv_folds,
  # metricas de evaluación
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE), # busqueda de hiperparaametros y las va a ir guardando
  grid = hiperpar_grid_3 # puedo usar de las que ya creamos arriba o poner un valor, usamos el 3 aue son pocas las combinaciones posibles
)

grid_fit |> unnest(.metrics) # para visualizar las metricas

# promedio de las metricas

grid_fit |>  collect_metrics(summarize = TRUE)

grid_fit |>  show_best(metric = "rmse", n = 5)

grid_fit |>  show_best(metric = "mae", n = 5)

# Evaluación en los diferenres hiperparaametros

# Métricas por separado
grid_fit |> collect_metrics(summarize = TRUE) |> 
  filter(.metric == "rmse") |> 
  pivot_longer(cols = c(tree_depth, min_n),
               values_to = "value",
               names_to = "parameter") |> 
  ggplot(aes(x=value, y=mean, color=parameter)) +
  geom_point() + geom_line() +
  facet_wrap(facets = vars(parameter), nrow = 2, scales = "free")

# Hiperparametros en conjunto
grid_fit |> collect_metrics(summarize = TRUE) |> 
  filter(.metric=="rmse") |> 
  ggplot(aes(x=tree_depth, y=min_n, color=mean, size=mean)) +
  geom_point() + 
  ggtitle("Evolución del error en función de los hiperparametros")

# Distribución de densidad del error
install.packages("ggpubr")
library(ggpubr)

p1 <- grid_fit |> collect_metrics(summarize = FALSE) |> 
  ggplot(aes(x=.metric, y=.estimate, fill=.metric, color=.metric)) +
  geom_boxplot() +
  coord_flip()

p2 <- grid_fit |> collect_metrics(summarize = FALSE) |> 
  ggplot(aes(x=.estimate, fill=.metric)) +
  geom_density(alpha=0.5)

# para juntar dos graficas pero con la libreria ggpurb
ggarrange(p1,p2,nrow=2)


# Elegir el mejor modelo
best_hip <- select_best(grid_fit, metric="rmse")
final_model_tree <- finalize_model(x= model_tree,
                                   parameters = best_hip)

final_model_tree

final_model_tree_fit <- final_model_tree |> 
  fit(formula = price ~.,
      data = data_train_prep)


# predicciones
#####incompleto

predicciones <- final_model_tree_fit |> 
  predict(new_data = data_test_prep,
          type = "numeric")

predicciones |> head()


prediccione <- prediccione |>  bind_cols(data_test_prep |> select(price))

rmse(predicciones, truth = price, estimate = .pred)

predicciones |>  ggplot(aes(x=price, y = .pred)) +
  geom_point() + 
  geom_abline(slope=1, intercept=0)