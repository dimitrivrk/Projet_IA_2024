import pandas as pd
import numpy as np
from predictionAge import load_and_preprocess_data
from json import loads, dumps
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = load_and_preprocess_data()
test_data = pd.concat(X_train)


def dico():
    # CREATION DU FICHIER PLK
    dico = {
        'or': OrdinalEncoder(categories=[['Jeune', 'Adulte', 'vieux', 'senescent']]),
        'oh': OneHotEncoder(sparse_output=False),
        'ss': StandardScaler(),
        'rf': RandomForestRegressor(n_estimators=200,
                                min_samples_split=2,
                                max_samples=0.8,
                                max_leaf_nodes=150,
                                max_depth=32,
                                random_state=42),
        # 'dt': dtr = DecisionTreeRegressor(max_depth=12,
        #                                 min_impurity_decrease=0.1,
        #                                 min_samples_leaf=3,
        #                                 min_samples_split=4),
        # 'ml': mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000),
        # 'pl': pls = PLSRegression(n_components=4),
    }
    with open('fichier_joblib.pkl', 'wb') as file:
        joblib.dump(dico, file)
        print("good")


def script(JSON_fichier, dico):

    new_data = pd.read_csv(JSON_fichier)
    print(new_data[['fk_nomtech']])

    dico = joblib.load('fichier_joblib.pkl')

    X = new_data[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'fk_nomtech', 'clc_nbr_diag']].copy()
    Y = new_data[['age_estim']]

    X[["fk_stadedev"]] = dico['or'].transform(new_data[["fk_stadedev"]])

    feuillage = dico['oh'].transform(X[["feuillage"]])
    feuillage_df = pd.DataFrame(feuillage, columns=dico['oh'].get_feature_names_out(["feuillage"]))
    #fk_nomtech = dico['oh'].transform(X[["fk_nomtech"]])
    #fk_nomtech_df = pd.DataFrame(fk_nomtech, columns=dico['oh'].get_feature_names_out(["fk_nomtech"]))


    X.drop(columns=["feuillage"], inplace=True)
    X = pd.concat([X, feuillage_df], axis=1)


    #X[["haut_tronc"]] = dico['ss'].fit_transform(X[["haut_tronc"]])
    #X[["tronc_diam"]] = dico['ss'].fit_transform(X[["tronc_diam"]])
    #X[["fk_stadedev"]] = dico['ss'].fit_transform(X[["fk_stadedev"]])
    X[["feuillage"]] = dico['ss'].transform(X[["feuillage"]])
    #X[["fk_nomtech"]] = dico['ss'].fit_transform(X[["fk_nomtech"]])
    #X[["clc_nbr_diag"]] = dico['ss'].fit_transform(X[["clc_nbr_diag"]])

    #X = dico['ss'].transform(X)



    print(dico['rf'])

    rfr_prediction = dico['rf'].predict(X_test)
    print(rfr_prediction)


script("Data_Arbre.csv", dico)
