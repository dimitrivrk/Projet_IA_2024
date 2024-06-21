from sklearn.model_selection import GridSearchCV
from predictionAge import load_and_preprocess_data, RandomForestRegressor, DecisionTreeRegressor, MLPRegressor, PLSRegression


# RANDOM FOREST
def grid_rfr(X_train, y_train, big=True):
    param_rfr_small = {
        'n_estimators': [200],
        'max_depth': [32],
        'max_leaf_nodes': [150],
        'max_samples': [0.8],
        'min_samples_split': [2, 100]
    }
    param_rfr_big = {
        'n_estimators': [50, 200, 500],
        'max_depth': [6, 18, 32],
        'max_leaf_nodes': [5, 50, 150],
        'max_samples': [0.1, 0.4, 0.8],
        'min_samples_split': [500, 1000, 2500]
    }
    param_rfr = param_rfr_big if big else param_rfr_small
    rfr = RandomForestRegressor()
    grid_search = GridSearchCV(rfr, param_rfr, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_dtrX_train(X_train, y_train):
    param_dtr = {
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4],
        'min_impurity_decrease': [0.3, 0.5, 0.8],
        'max_depth': [3,  10, 15, 20, 30]
    }
    dtr = DecisionTreeRegressor()
    grid_search = GridSearchCV(dtr, param_dtr, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_mlp(X_train, y_train):
    param_mlp = {
        'hidden_layer_sizes': [(50, 50), (100,)],
        'activation': ["identity", "logistic", "tanh", "relu"],
        'solver': ["lbfgs", "sgd", "adam"],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 800]
    }
    mlp = MLPRegressor()
    grid_search = GridSearchCV(mlp, param_mlp, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_pls(X_train, y_train):
    param_pls = {
        'n_components': [1,2, 3, 4, 5, 6],
        'scale': [True, False],
        'max_iter': [2, 4, 5, 6, 7, 8]
    }
    pls = PLSRegression()
    grid_search = GridSearchCV(pls, param_pls, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Meilleurs params:", grid_search.best_params_)
    print("Meilleur score:", grid_search.best_score_)


def grid_search_all(X_train, y_train):
    grid_rfr(X_train, y_train)
    grid_dtrX_train(X_train, y_train)
    grid_mlp(X_train, y_train)
    grid_pls(X_train, y_train)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    grid_search_all(X_train, y_train)
