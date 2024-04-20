library(rpart)
library(rpart.plot)
library(tidymodels)
library(tidyverse)
library(mlbench)
# install.packages("mlbench")


data("PimaIndiansDiabetes2", package = "mlbench")

data <- PimaIndiansDiabetes2
summary(data)


# Revisar si hay datos faltantes

library(DataExplorer)
plot_missing(data)

# removiendo las columnas que tienen muchos na
data <- data |> select(-c(insulin, triceps))
data <- na.omit(data)

plot_missing(data)

# Observar las personas que se emabarazon y que tienen diabetes

data |>
  ggplot(aes(diabetes, y= pregnant)) +
  geom_boxplot()

# revisar la variable objetivo

class(data$diabetes)
levels(data$diabetes)

data$diabetes <- relevel(data$diabetes, ref = "pos") # pos variable de relevancia

set.seed(123)

split_initial <- initial_split(data = data,
                               prop = .7, # training
                               strata = diabetes)
train <- training(split_initial)
test <- testing(split_initial)

# Balance checar si esta balanceado el split para no hacer oversampling los datos deben de estar a lo mucho 35%

 data |> 
   count(diabetes) |> 
   mutate(balance=n/sum(n))
 
 # Modelo de decision
 # full tree
 
 model_full_tree <- rpart(diabetes ~ .,
       data = train) # empezar a podar agregando el parametro minsplit
 
 model_full_tree
 
 rpart.plot(model_full_tree)
 
 # prediccion
 # performance
 
 
 
 # modelo decision tree
 # prune tree
 
 # correr el arbol completo
 model_full_tree
 rpart.plot(model_full_tree)
 
 printcp(model_full_tree) # cp es parametro de complejidad, el alpha
 


 