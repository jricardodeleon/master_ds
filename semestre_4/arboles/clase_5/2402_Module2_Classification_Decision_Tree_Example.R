library(rpart)
library(rpart.plot)
library(readxl)


#-------------------------------------------------------------
#Example 2
getwd()
data2 <- readxl::read_excel("/Users/macbookpro/Documents/master/master_ds/semestre_4/arboles/clase_5/2403_Classification_Decision_Tree_Example2.xlsx")
data2$married <- as.factor(data2$married) # especificamos que es variable categorica
data2$graduate <- as.factor(data2$graduate)
data2$loan <- as.factor(data2$loan)
data2$invest <- as.factor(data2$invest)

model2 <- rpart(formula = invest ~ .,
               data = data2)
model2

rpart.plot(model2)




  
