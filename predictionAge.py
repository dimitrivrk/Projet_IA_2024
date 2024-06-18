import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV



#* remplacer les NA de age_estim a l'aide d'une regression lineaire entre les age_estim connus et leur tronc_diam connus de la meme espece qui determine la valeur de age_estim pour un tronc_diam donné
#? OK!


# Function to replace missing age_estim values
#replace_missing_age_estim <- function(data) {
  #for (i in seq_len(nrow(data))) {
    #if (is.na(data[i, "age_estim"])) {
      #vector <- c()
      #for (j in c('tronc_diam', 'haut_tronc', 'fk_stadedev','feuillage')) {
        #if (!is.na(data[i, j])) {
          #vector <- c(vector, j)
        #}
      #}
      #if (length(vector) > 0) {
        # Vérifier et exclure les prédicteurs colinéaires
        #formula <- as.formula(paste("age_estim ~", paste(vector, collapse = " + ")))
        #model <- lm(formula, data = data, na.action = na.exclude)
        #new_data <- data.frame(
          #tronc_diam = data[i, "tronc_diam"],
          #haut_tronc = data[i, "haut_tronc"],
          #fk_stadedev = data[i, "fk_stadedev"],
          #feuillage = data[i, "feuillage"]

        #)
        # Garder seulement les colonnes présentes dans le modèle
        #new_data <- new_data[, vector, drop = FALSE]
        #predicted_value <- predict(model, newdata = new_data)
        #data[i, "age_estim"] <- max(round(predicted_value), 0)
      #}
    #}
  #}
  #return(data)
#}

#cat("Nombre de valeurs manquantes avant : ", sum(is.na(data$age_estim)), "\n")
#data <- replace_missing_age_estim(data)
#cat("Nombre de valeurs manquantes après : ", sum(is.na(data$age_estim)), "\n")

#data <- data[!is.na(data$age_estim), ]

datarbre = pd.read_csv("Data_Arbre.csv")

tronc_diam = datarbre[["tronc_diam"]]
haut_tronc = datarbre[["haut_tronc"]]
fk_stadedev = datarbre[["fk_stadedev"]]
feuillage = datarbre[["feuillage"]]
age_estim = datarbre[["age_estim"]]
fk_nomtech = datarbre[["fk_nomtech"]]
clc_nbr_diag = datarbre[["clc_nbr_diag"]]

#ENCODING
ordinal_encoder = OrdinalEncoder()
fk_stadedev_enc = ordinal_encoder.fit_transform(fk_stadedev)
feuillage_enc = ordinal_encoder.fit_transform(feuillage)
fk_nomtech_enc = ordinal_encoder.fit_transform(fk_nomtech)

#NORMALISATING
ss = StandardScaler()
haut_tronc_nrm = ss.fit_transform(haut_tronc)
fk_stadedev_nrm = ss.fit_transform(fk_stadedev_enc)
feuillage_nrm = ss.fit_transform(feuillage_enc)
tronc_diam_nrm = ss.fit_transform(tronc_diam)
age_estim_nrm = ss.fit_transform(age_estim)
fk_nomtech_nrm = ss.fit_transform(fk_nomtech_enc)

datarbre[["fk_stadedev"]] = fk_stadedev_enc
datarbre[["feuillage"]] = feuillage_enc

X = datarbre[['tronc_diam', 'haut_tronc', 'fk_stadedev', 'feuillage']]
Y = datarbre['age_estim']

#SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, Y , train_size = 0.8, test_size = 0.2, random_state = 42)

#RANDOM FOREST
rfr = RandomForestRegressor(n_estimators = 200, min_samples_split = 2, max_samples = 0.8, max_leaf_nodes = 150, max_depth = 32, random_state=42)
rfr.fit(X_train, y_train)
prediction = rfr.predict(X_test)

precision = r2_score(y_test, prediction)
#print("\nRandomForest : ", precision)

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
#print("Rmse : ", rmse)

mae = mean_absolute_error(y_test, prediction)
#print("Mae : ", mae)

#CART
dtr = DecisionTreeRegressor(max_depth=12, min_impurity_decrease=0.1, min_samples_leaf=3, min_samples_split=4)
dtr.fit(X_train, y_train)
prediction2 = dtr.predict(X_test)

precision = r2_score(y_test, prediction2)
print("\nCART : ", precision)

mse = mean_squared_error(y_test, prediction2)
rmse = np.sqrt(mse)
print("Rmse : ", rmse)

mae = mean_absolute_error(y_test, prediction2)
print("Mae : ", mae)

#NEURONES
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
mlp.fit(X_train, y_train)
prediction3 = mlp.predict(X_test)

precision = r2_score(y_test, prediction3)
#print("\nNEURONES : ", precision)

mse = mean_squared_error(y_test, prediction3)
rmse = np.sqrt(mse)
#print("Rmse : ", rmse)

mae = mean_absolute_error(y_test, prediction3)
#print("Mae : ", mae)

#PLS
pls = PLSRegression(n_components=4)
pls.fit(X_train, y_train)
prediction4 = pls.predict(X_test)

precision = r2_score(y_test, prediction4)
#print("\nPLS : ", precision)

mse = mean_squared_error(y_test, prediction4)
rmse = np.sqrt(mse)
#print("Rmse : ", rmse)

mae = mean_absolute_error(y_test, prediction4)
#print("Mae : ", mae)

#GRID SEARCH
#RANDOM FOREST
#param_rfr = {
    #'n_estimators': [10, 50, 200, 500],
    #'max_depth': [2, 6, 18, 32],
    #'max_leaf_nodes': [5, 25, 75, 150],
    #'max_samples': [0.1, 0.4, 0.8],
    #'min_samples_split': [200, 500, 1000, 2500, 4000],
#}

#CART
#param_dtr = {
    #'min_samples_split': [1, 2, 3, 4],
    #'min_samples_leaf': [1, 2, 3, 4],
    #'min_impurity_decrease': [0.1, 0.3, 0.5, 0.8],
    #'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
#}

#MLP
#param_mlp = {
    #'hidden_layer_sizes': [(50,50), (100,)],
    #'activation': ["identity", "logistic", "tanh", "relu"],
    #'solver': ["lbfgs", "sgd", "adam"],
    #'learning_rate': ['constant', 'adaptive'],
    #'max_iter': [500, 800]
#}

#grid_search = GridSearchCV(dtr, param_dtr, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
#grid_search.fit(X_train, y_train)
#print("Meilleure profondeur:", grid_search.best_params_)
#print("Meilleur score:", grid_search.best_score_)