#Saratoga Houses

#16 Variables

#price: Price (US dollars)
#lotSize: Size of lot (acres)
#age: age of house (years)
#landValue: value of land (US dollars)
#livingArea: living are (square feet)
#pctCollege: percent of neighborhood that graduated college
#bedrooms: number of bedrooms
#fireplaces: number of fireplaces
#bathrooms: number of bathrooms (half bathrooms have no shower or tub)
#Rooms: number of rooms
#heating: Type of heating system
#fuel: fuel used for heating
#sewer: type of sewer system
#waterfront: whether property includes waterfront
#newConstruction: whether the property is a new construction
#centralAir: whether the house has central air

#install.packages("mosaicData")
#install.packages("tidymodels")
library(mosaicData)
library(tidymodels)
library(tidyverse)

data("SaratogaHouses")
data <- SaratogaHouses

colnames(data)

summary(data)

#-------------------------------------------------------------------------
#SPLIT TRAIN / TEST
#RSAMPLE

set.seed(123)

split_incial <- initial_split(data= data,
              prop= 0.8, #training proportion
              strata = price #target variable
              )

data_train <- training(split_incial)
data_test <- testing(split_incial)

summary(data_train$price)
summary(data_test$price)

#--------------------------------------------------------------------------
#PRE-PROCESSING
#Recipes

#Set Transformations
#recipe()
transformer <- recipe(formula= price ~.,
       data=data_train) %>% 
      #step_mutate( value_per_acres = landValue/lotSize) %>% 
      step_naomit(all_predictors()) %>% 
      step_nzv (all_predictors()) %>% 
      step_center(all_numeric(),-all_outcomes()) %>% 
      step_scale(all_numeric(),-all_outcomes()) %>% 
      step_dummy(all_nominal(),-all_outcomes()) 

transformer      

#Training recipe
#prep()

transformer_fit <- prep(transformer)

#Apply transformations to datasets
#bake()

data_train_prep <- bake(transformer_fit,new_data = data_train)
data_test_prep <- bake(transformer_fit,new_data = data_test)

#---------------------------------------------------
#MODEL
#Choose model
#Define the hyper parameter to optimize

model_tree <- decision_tree(mode = "regression", #regression, classification
              tree_depth = tune(),
              min_n = tune()) %>% 
      set_engine(engine = "rpart") #paqueteria

#Cross Validation
cv_folds <- vfold_cv(data = data_train,
         v = 5, #numero de particiones
         strata = price #variable objetivo
         )

#-------------------------------------------------------------------------
#OPTIMIZATION 
#tune()

#Randoom Grid
hiperpar_grid_1 <-  grid_random(
  #Rango de busqueda de cada hiperparametro
  tree_depth(range = c(1,10),trans = NULL),
  min_n( range = c(2,50),trans = NULL ),
  size = 100 #Numero de combinaciones aleatorias a probar
)

#Grid search (Example)
hiperpar_grid_2 <- grid_regular(
  #Rango de busqueda de cada hiperparametro
  tree_depth(range = c(1,10),trans = NULL),
  min_n( range = c(2,50),trans = NULL ),
  #numero de valores para cada hiperametro
  levels = c(10,49)
)

#Grid search (Example)
hiperpar_grid_3 <- grid_regular(
  #Rango de busqueda de cada hiperparametro
  tree_depth(range = c(1,10),trans = NULL),
  min_n( range = c(2,50),trans = NULL ),
  #numero de valores para cada hiperametro
  levels = c(10,10)
)

#Optimization
grid_fit <- tune_grid(
  object = model_tree,
  #Pre procesamiento de los datos
  #Recipe
  preprocessor = transformer,
  #Resample que se crearon sin preprocesar
  resamples = cv_folds,
  #Metricas evaluacion
  metrics = metric_set(rmse,mae),
  control = control_grid(save_pred = TRUE),
  grid = hiperpar_grid_3
)


#-------------------------------------------------------------------------
#EVALUATION METRICS
#tune

grid_fit

grid_fit %>% unnest(.metrics)

grid_fit %>% collect_metrics(summarize = TRUE)

grid_fit %>% show_best(metric = "rmse",n=5)

grid_fit %>% show_best(metric = "mae",n=5)


#Grafica
#Evolucion del error

#Hiperparametros por separado
grid_fit %>% collect_metrics(summarize = TRUE) %>% 
  filter(.metric=="rmse") %>% 
  pivot_longer(cols = c(tree_depth,min_n),
               values_to = "value",
               names_to = "parameter") %>% 
  ggplot(aes(x=value,y=mean,color=parameter)) +
  geom_point() + geom_line() +
  facet_wrap(facets = vars(parameter),nrow=2,scales = "free")

#Hiperparametros en conjunto
grid_fit %>% collect_metrics(summarize = TRUE) %>% 
  filter(.metric=="rmse") %>%
  ggplot(aes(x=tree_depth,y=min_n,color=mean,size=mean)) +
  geom_point() +
  ggtitle("Evolucion del error en función de los hiperparametros")
  

library(ggpubr)

p1 <- grid_fit %>% collect_metrics(summarize = FALSE) %>% 
  ggplot(aes(x=.metric,y=.estimate,fill=.metric,color=.metric)) +
  geom_boxplot() +
  coord_flip()

p2 <-  grid_fit %>% collect_metrics(summarize = FALSE) %>% 
  ggplot(aes(x=.estimate,fill=.metric)) +
  geom_density(alpha=0.5) 

ggarrange(p1,p2,nrow=2)

#Choose the best model

best_hip <- select_best(grid_fit,metric="rmse")

final_model_tree <- finalize_model(x=model_tree,
                                   parameters = best_hip )

final_model_tree

final_model_tree_fit <- final_model_tree %>% 
  fit(formula = price ~.,
      data = data_train_prep)

#-------------------------------------------------------------------------
#PREDICTION

predicciones <- final_model_tree_fit %>% 
  predict(new_data = data_test_prep,
          type = "numeric")

predicciones %>%  head()
    
predicciones <- predicciones %>% bind_cols(data_test_prep %>% select(price))  

rmse(predicciones,truth = price, estimate= .pred)


#Plot prediction & residuals

p1 <- predicciones %>%  ggplot(
  aes(x = price, y = .pred)
) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "firebrick") +
  labs(title = "Valor predicho vs valor real") +
  theme_bw()

p2 <- predicciones %>% ggplot(
  aes(x = price - .pred)
) +
  geom_density() + 
  labs(title = "Distribución residuos del modelo") +
  theme_bw()

ggarrange(plotlist = list(p1, p2)) %>%
  annotate_figure(
    top = text_grob("Distribución residuos", size = 15, face = "bold")
  )


















