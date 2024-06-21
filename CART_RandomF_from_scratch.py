import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
data = pd.read_csv("Data_Arbre.csv")

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CART:
    def __init__(self, min_samples_split=2, max_depth=2, task="regression"):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.task = task
        
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["var_red"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree)
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red > max_var_red:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "var_red": curr_var_red
                        }
                        max_var_red = curr_var_red
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        left_mask = dataset[:, feature_index] <= threshold
        right_mask = ~left_mask
        return dataset[left_mask], dataset[right_mask]
    
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if self.task == "regression":
            reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        else:
            reduction = self.gini(parent) - (weight_l * self.gini(l_child) + weight_r * self.gini(r_child))
        return reduction
    
    def gini(self, y):
        m = len(y)
        return 1.0 - np.sum((np.bincount(y) / m) ** 2)
    
    def calculate_leaf_value(self, Y):
        if self.task == "regression":
            return np.mean(Y)
        else:
            return np.argmax(np.bincount(Y))
                
    def fit(self, X, Y):
        dataset = np.hstack((X, Y.reshape(-1, 1)))
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        return np.array([self.make_prediction(x, self.root) for x in X])
    
    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        if x[tree.feature_index] <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Préparation des données
data_learning = data[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'feuillage', 'fk_nomtech', 'clc_nbr_diag']].copy()

# ENCODAGE
ordinal_encoder = OrdinalEncoder(categories=[['Jeune', 'Adulte', 'vieux', 'senescent']])
onehot_encoder = OneHotEncoder(sparse_output=False)  # Utilisation de sparse_output

data_learning["fk_stadedev"] = ordinal_encoder.fit_transform(data_learning[["fk_stadedev"]])

feuillage_encoded = onehot_encoder.fit_transform(data_learning[["feuillage"]])
feuillage_encoded_df = pd.DataFrame(feuillage_encoded, columns=onehot_encoder.get_feature_names_out(input_features=["feuillage"]))

nomtech_encoded = onehot_encoder.fit_transform(data_learning[["fk_nomtech"]])
nomtech_encoded_df = pd.DataFrame(nomtech_encoded, columns=onehot_encoder.get_feature_names_out(input_features=["fk_nomtech"]))

# Mise à jour du dataframe
data_learning.drop(columns=["feuillage", "fk_nomtech"], inplace=True)
data_learning = pd.concat([data_learning, feuillage_encoded_df, nomtech_encoded_df], axis=1)

# NORMALISATION
ss = StandardScaler()
data_learning[["haut_tronc", "tronc_diam"]] = ss.fit_transform(data_learning[["haut_tronc", "tronc_diam"]])

X = data_learning.values
Y = data["age_estim"].values

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

# Modèle de régression
regressor = CART(min_samples_split=3, max_depth=3, task="regression")
regressor.fit(X_train, y_train)

Y_pred = regressor.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, Y_pred)))
print("R2 Score:", r2_score(y_test, Y_pred))

# Modèle Random Forest
class RandomForest:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2, task="regression"):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.task = task
        self.trees = []
        
    def fit(self, X, Y):
        for _ in range(self.n_estimators):
            tree = CART(self.min_samples_split, self.max_depth, self.task)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_X = X[bootstrap_indices]
            bootstrap_Y = Y[bootstrap_indices]
            tree.fit(bootstrap_X, bootstrap_Y)
            self.trees.append(tree)
            
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "regression":
            return np.mean(predictions, axis=0)
        else:
            return np.round(np.mean(predictions, axis=0))

rf_regressor = RandomForest(n_estimators=5, min_samples_split=3, max_depth=3, task="regression")
rf_regressor.fit(X_train, y_train)
Y_pred = rf_regressor.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, Y_pred)))
print("R2 Score:", r2_score(y_test, Y_pred))
