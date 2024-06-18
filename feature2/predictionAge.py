import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

datarbre = pd.read_csv("Data_Arbre.csv")
data_learning = datarbre[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'fk_nomtech', 'clc_nbr_diag']].copy()

#ENCODING
ordinal_encoder = OrdinalEncoder(categories=[['Jeune', 'Adulte', 'vieux', 'senescent']])
onehot_encoder = OneHotEncoder()

data_learning[["fk_stadedev"]] = ordinal_encoder.fit_transform(data_learning[["fk_stadedev"]])
data_learning[["feuillage"]] = onehot_encoder.fit_transform(data_learning[["feuillage"]])
data_learning[["fk_nomtech"]] = onehot_encoder.fit_transform(data_learning[["fk_nomtech"]])

#NORMALISATING
ss = StandardScaler()
# apply the standardization on the data_learning
data_learning[["haut_tronc"]] = ss.fit_transform(data_learning[["haut_tronc"]])
data_learning[["tronc_diam"]] = ss.fit_transform(data_learning[["tronc_diam"]])

X = data_learning
Y = datarbre["age_estim"]

#SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

#RANDOM FOREST
rfr = RandomForestRegressor(n_estimators=200,
                            min_samples_split=2,
                            max_samples=0.8,
                            max_leaf_nodes=150,
                            max_depth=32,
                            random_state=42)
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
