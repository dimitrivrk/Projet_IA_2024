from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import numpy as np
import seaborn as sns


def anomaly_detection_OneClassSVM(df_learning, nu=0.05) -> 'indexes':
    """
    This function detects anomalies in the data using the One-Class SVM algorithm
    :param df_learning: the data frame with the selected columns for learning
    :param nu: the proportion of outliers
    :return: the indexes of the anomalies
    """
    anomaly_detector = OneClassSVM(kernel='rbf', nu=nu)
    result = anomaly_detector.fit_predict(df_learning)
    anomalies_indexes = np.where(result == -1)

    return anomalies_indexes


def anomaly_detection_IsolationForest(df_learning, contamination=0.01, bootstrap=False) -> 'indexes':
    """
    This function detects anomalies in the data using the Isolation Forest algorithm
    :param bootstrap: True or False
    :param df_learning: the data frame with the selected columns for learning
    :param contamination: the threshold for the decision function
    :return: the indexes of the anomalies
    """
    anomaly_detector = IsolationForest(bootstrap=bootstrap)
    result = anomaly_detector.fit_predict(df_learning)
    anomalies_indexes = np.where(result == -1)

    return anomalies_indexes


def plot_anomalies(df_learning, contamination=0.1):
    model_IF = IsolationForest(contamination=contamination)
    model_IF.fit(df_learning)
    df_learning['anomaly'] = model_IF.predict(df_learning)
    palette = ['orange', 'blue']
    sns.pairplot(df_learning, vars=['haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_stadedev', 'fk_port'], hue='anomaly', palette=palette)
    plt.suptitle('Anomaly Detection Using Isolation Forest\n'
                 'number of anomalies: ' + str(len(df_learning[df_learning['anomaly'] == -1])),
                 y=0.95, fontsize=10)
    plt.show()
