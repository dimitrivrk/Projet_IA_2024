from sklearn.model_selection import GridSearchCV
from predictionAge import load_and_preprocess_data, RandomForestRegressor, DecisionTreeRegressor, MLPRegressor, PLSRegression


# GRID SEARCH
# RANDOM FOREST
def grid_rfr(X_train, y_train):
    param_rfr = {
        'n_estimators': [10, 50, 200, 500],
        'max_depth': [2, 6, 18, 32],
        'max_leaf_nodes': [5, 25, 75, 150],
        'max_samples': [0.1, 0.4, 0.8],
        'min_samples_split': [200, 500, 1000, 2500, 4000],
    }
    rfr = RandomForestRegressor()
    grid_search = GridSearchCV(rfr, param_rfr, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


# CART
def grid_dtrX_train(X_train, y_train):
    param_dtr = {
        'min_samples_split': [1, 2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4],
        'min_impurity_decrease': [0.1, 0.3, 0.5, 0.8],
        'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    }
    dtr = DecisionTreeRegressor()
    grid_search = GridSearchCV(dtr, param_dtr, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


# MLP
def grid_mlp(X_train, y_train):
    param_mlp = {
        'hidden_layer_sizes': [(50, 50), (100,)],
        'activation': ["identity", "logistic", "tanh", "relu"],
        'solver': ["lbfgs", "sgd", "adam"],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 800]
    }
    mlp = MLPRegressor()
    grid_search = GridSearchCV(mlp, param_mlp, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_pls(X_train, y_train):
    param_pls = {
        'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'scale': [True, False],
        'max_iter': [100, 200, 300, 400, 500]
    }
    pls = PLSRegression()
    grid_search = GridSearchCV(pls, param_pls, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_search_all(X_train, y_train):
    grid_rfr(X_train, y_train)
    grid_dtrX_train(X_train, y_train)
    grid_mlp(X_train, y_train)
    grid_pls(X_train, y_train)


def cor_age_diam(data):
    filtered_data_tronc_diam = data[data["tronc_diam"] != 0]
    correlation_value_tronc_diam = filtered_data_tronc_diam[["age_estim", "tronc_diam"]].corr().iloc[0, 1]
    print(correlation_value_tronc_diam)
    correlation_matrix_tronc_diam = filtered_data_tronc_diam[["age_estim", "tronc_diam"]].corr()
    print(correlation_matrix_tronc_diam)

#cor_age_diam(datarbre)

#print(datarbre[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'fk_nomtech', 'clc_nbr_diag', 'age_estim']].corr())

#grid_rfr(X_train, y_train)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    grid_search_all(X_train, y_train)