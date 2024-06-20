import pandas as pd
import numpy as np
from predictionAge import load_and_preprocess_data
from json import loads, dumps
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = load_and_preprocess_data()


def dico():
    # CREATION DU FICHIER PLK
    dico = {
        'or': ordinal_encoder,
        'oh': onehot_encoder,
        'ss': ss,
        'rf': rfr,
        # 'dt': dtr,
        # 'ml': mlp,
        # 'pl': pls,
    }
    with open('fichier_joblib.pkl', 'wb') as file:
        joblib.dump(dico, file)
        print("good")


def script(JSON_fichier, dico):

    new_data = pd.read_csv(JSON_fichier)
    print(new_data['fk_nomtech'])

    dico = joblib.load('fichier_joblib.pkl')

    new_data[["fk_stadedev"]] = dico['or'].transform(new_data[["fk_stadedev"]])
    new_data[["fk_nomtech"]] = dico['oh'].transform(new_data[["fk_nomtech"]])

    new_data = dico['ss'].transform(new_data)

    X = new_data[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'fk_nomtech', 'clc_nbr_diag']]
    Y = new_data[['age_estim']]

    print(dico['rf'])

    rfr_precision = r2_score(y_test, rfr_prediction)
    rfr_rmse = np.sqrt(mean_squared_error(y_test, rfr_prediction))
    rfr_mae = mean_absolute_error(y_test, rfr_prediction)


script("Data_Arbre.csv", dico)
