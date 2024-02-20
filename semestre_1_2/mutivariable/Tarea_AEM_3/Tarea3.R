library(tidyverse)
library(tidymodels)
library(factoextra)
library(NbClust)
library(funModeling)
library(plotly)
library(ggplot2)
library(gt)
library(cluster)
library(RColorBrewer)

# Load File
df <- read_csv("/Users/khrw896/Documents/Master/mutivariable/Tarea3/Wholesale customers data.csv", show_col_types = FALSE)

# show the data frame content
glimpse(df)

# show the first 10 rows of elements
head(df, 10)

# show data summary
summary(df)

# filter data and scale
df_filtered <- df %>% select(Fresh, Delicatessen) %>% scale() %>% as.data.frame()

# plot data set
ggplot() + geom_point(aes(x = Fresh, y = Delicatessen), data = df_filtered, alpha = 0.5) + ggtitle('Conjunto de Datos')

# Elbow Method
fviz_nbclust(x = df_filtered, kmeans, method = "wss", k.max = 12) +
  labs(title = "Número óptimo de clusters")

# Calcular tot withhinss
calcular_totwithinss <- function(n_clusters, df_filtered, iter.max=1000, nstart=25){
  # Esta función aplica el algoritmo kmeans y devuelve la suma total de
  # cuadrados internos.
  cluster_kmeans <- kmeans(centers = n_clusters, x = df_filtered, iter.max = iter.max,
                           nstart = nstart)
  return(cluster_kmeans$tot.withinss)
}

# Se aplica esta función con para diferentes valores de k
total_withinss <- map_dbl(.x = 1:12,
                          .f = calcular_totwithinss,
                          df_filtered = df_filtered)
total_withinss

data.frame(n_clusters = 1:12, suma_cuadrados_internos = total_withinss) %>%
  ggplot(aes(x = n_clusters, y = suma_cuadrados_internos)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:12) +
  labs(title = "Evolución de la suma total de cuadrados intra-cluster") +
  theme_bw()


# Se puede obtener el el estadístico gap con la función fviz_nbclust() o con la función clusGap() del paquete cluster.

set.seed(896)
kmeans_gap <- clusGap(x = df_filtered, 
                      FUNcluster = kmeans,
                      K.max = 12,
                      B = 150,
                      verbose = FALSE,
                      nstart = 25)
kmeans_gap


print(kmeans_gap, method="globalmax")


# Center each value
kmeans_mutated <- tibble(k = 1:12) %>%
  mutate(kmeans = map(k, ~kmeans(df_filtered, centers = .x)),
         kmeans_tidy = map(kmeans, tidy),
         kmeans_glan = map(kmeans, glance),
         kmeans_augm = map(kmeans, augment, df_filtered))


clusters_mutated <- kmeans_mutated %>%
  unnest(cols = c(kmeans_tidy))

clusters_mutated %>%
  select(k, Fresh, Delicatessen, size, withinss, cluster) %>%
  head(20) %>%
  gt() %>%
  tab_header(title = md("Detalles por Cluster"))


data_assigned <- kmeans_mutated %>%
  unnest(cols = c(kmeans_augm))

data_assigned %>%
  select(k, Fresh, Delicatessen , .cluster)

graph <- ggplot(data_assigned, aes(x = Fresh, y = Delicatessen)) +
  geom_point(aes(color = .cluster), alpha = 0.8) + 
  facet_wrap(~ k)
theme(legend.position = "right")

graph + geom_point(data = clusters_mutated, size = 6, shape = "*")

# another way to plot the values in the same graph

set.seed(896)
kmeans <- kmeans(df_filtered, 12, iter.max = 1000, nstart = 10)
kmeans$centers

df_filtered$cluster <- kmeans$cluster
ggplot() + geom_point(aes(x = Fresh, y = Delicatessen, color = cluster), data = df_filtered, size = 2) +
  scale_colour_gradientn(colours=rainbow(4)) +
  geom_point(aes(x = kmeans$centers, y = kmeans$center), color = 'black', size = 3) + 
  ggtitle('Clusters de Datos con k = 12 / K-Medios') + 
  xlab('Fresh') + ylab('Delicatessen')

