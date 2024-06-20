import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression


def load_and_preprocess_data(split=True):
    datarbre = pd.read_csv("Data_Arbre.csv")
    data_learning = datarbre[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'clc_nbr_diag']].copy()

    # ENCODING
    ordinal_encoder = OrdinalEncoder(categories=[['Jeune', 'Adulte', 'vieux', 'senescent']])
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output

    data_learning["fk_stadedev"] = ordinal_encoder.fit_transform(data_learning[["fk_stadedev"]])

    # Fit and transform the one-hot encoder for each categorical feature
    feuillage_encoded = onehot_encoder.fit_transform(data_learning[["feuillage"]])
    feuillage_encoded_df = pd.DataFrame(feuillage_encoded, columns=onehot_encoder.get_feature_names_out(["feuillage"]))

    # Drop the original categorical columns and concatenate the encoded columns
    data_learning.drop(columns=["feuillage"], inplace=True)
    data_learning = pd.concat([data_learning, feuillage_encoded_df], axis=1)

    # NORMALISATING
    ss = StandardScaler()
    # apply the standardization on the data_learning
    data_learning[["haut_tronc"]] = ss.fit_transform(data_learning[["haut_tronc"]])
    data_learning[["tronc_diam"]] = ss.fit_transform(data_learning[["tronc_diam"]])

    if not split:
        return pd.concat([data_learning, datarbre["age_estim"]], axis=1)

    X = data_learning
    Y = datarbre["age_estim"]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # RANDOM FOREST
    rfr = RandomForestRegressor(n_estimators=200,
                                min_samples_split=2,
                                max_samples=0.8,
                                max_leaf_nodes=150,
                                max_depth=32,
                                random_state=42)
    rfr.fit(X_train, y_train)
    rfr_prediction = rfr.predict(X_test)
    print(rfr_prediction)
    rfr_precision = r2_score(y_test, rfr_prediction)
    rfr_rmse = np.sqrt(mean_squared_error(y_test, rfr_prediction))
    rfr_mae = mean_absolute_error(y_test, rfr_prediction)
    print(f"Random Forest :\nprécision={rfr_precision}\nrmse={rfr_rmse}\nmae={rfr_mae}\n")

    # CART
    dtr = DecisionTreeRegressor(max_depth=12,
                                min_impurity_decrease=0.1,
                                min_samples_leaf=3,
                                min_samples_split=4)
    dtr.fit(X_train, y_train)
    dtr_prediction = dtr.predict(X_test)

    dtr_precision = r2_score(y_test, dtr_prediction)
    dtr_rmse = np.sqrt(mean_squared_error(y_test, dtr_prediction))
    dtr_mae = mean_absolute_error(y_test, dtr_prediction)
    print(f"CART :\nprécision={dtr_precision}\nrmse={dtr_rmse}\nmae={dtr_mae}\n")

    # NEURONES
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    mlp.fit(X_train, y_train)
    mlp_prediction = mlp.predict(X_test)

    mlp_precision = r2_score(y_test, mlp_prediction)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_prediction))
    mlp_mae = mean_absolute_error(y_test, mlp_prediction)
    print(f"neuronnes (MLP):\nprécision={mlp_precision}\nrmse={mlp_rmse}\nmae={mlp_mae}\n")

    #PLS
    pls = PLSRegression(n_components=4)
    pls.fit(X_train, y_train)
    pls_prediction = pls.predict(X_test)

    pls_precision = r2_score(y_test, pls_prediction)
    pls_rmse = np.sqrt(mean_squared_error(y_test, pls_prediction))
    pls_mae = mean_absolute_error(y_test, pls_prediction)
    print(f"PLS:\nprécision={pls_precision}\nrmse={pls_rmse}\nmae={pls_mae}\n")
