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


def export_data_frame2JSON(dataframe, filename='Data_Arbre.json'):
    """
    Export a dataframe to a JSON file
    :param dataframe: the dataframe to export
    :param filename: the name of the file
    """
    dataframe.to_json(filename)


def export_models():
    """
    store the encoders and regressors in a dictionary and save it in a file
    """
    datarbre = pd.read_csv("Data_Arbre.csv")

    ordinal_encoder = OrdinalEncoder(categories=[['Jeune', 'Adulte', 'vieux', 'senescent']])
    ordinal_encoder.fit(datarbre[["fk_stadedev"]])

    one_hot_encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output
    one_hot_encoder.fit(datarbre[["feuillage"]])

    standard_scaler = StandardScaler()
    standard_scaler.fit(datarbre[["haut_tronc", "tronc_diam"]])

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    rfr = RandomForestRegressor(n_estimators=200,
                                min_samples_split=2,
                                max_samples=0.8,
                                max_leaf_nodes=150,
                                max_depth=32,
                                random_state=42)
    rfr.fit(X_train, y_train)

    dtr = DecisionTreeRegressor(max_depth=12,
                                min_impurity_decrease=0.1,
                                min_samples_leaf=3,
                                min_samples_split=4)
    dtr.fit(X_train, y_train)

    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    mlp.fit(X_train, y_train)

    pls = PLSRegression(n_components=4)
    pls.fit(X_train, y_train)

    dico = {
        'ordinal_encoder': ordinal_encoder,
        'one_hot_encoder': one_hot_encoder,
        'standard_scaler': StandardScaler(),
        'rfr': rfr,
        'dtr': dtr,
        'mlr': mlp,
        'plr': pls
    }

    with open('fichier_joblib.pkl', 'wb') as file:
        joblib.dump(dico, file)
        print("encoders and regressors saved in fichier_joblib.pkl")


def script(JSON_filename, MODELS_filename='fichier_joblib.pkl'):
    """
    Load the data from the JSON file, preprocess it and predict the age of the trees
    :param JSON_filename: the file that store a dataframe in JSON format
    :param MODELS_filename: the pickle file that store the encoders and regressors
    :return: a data frame with only one column : 'age_pred' in a JSON file
    """
    datarbre = pd.read_csv(JSON_filename)
    models = joblib.load('fichier_joblib.pkl')

    X = datarbre[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'clc_nbr_diag']].copy()
    Y = datarbre[['age_estim']]

    # encoding
    X[["fk_stadedev"]] = models['ordinal_encoder'].transform(X[["fk_stadedev"]])

    feuillage = models['one_hot_encoder'].transform(X[["feuillage"]])
    feuillage_df = pd.DataFrame(feuillage, columns=models['one_hot_encoder'].get_feature_names_out(["feuillage"]))
    X.drop(columns=["feuillage"], inplace=True)
    X = pd.concat([X, feuillage_df], axis=1)

    # normalizing
    X[['haut_tronc', 'tronc_diam']] = models['standard_scaler'].fit_transform(X[['haut_tronc', 'tronc_diam']])

    # predict the age of the trees
    age_predicted = models['rfr'].predict(X)

    # export age_predicted to a JSON file
    pd.DataFrame(age_predicted, columns=['age_pred']).to_json('age_predicted.json')


if __name__ == '__main__':
    script("Data_Arbre.csv")

    age_pred = pd.read_json('age_predicted.json')
    print(age_pred.head(10))
