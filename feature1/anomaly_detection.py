from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import numpy as np
import seaborn as sns


# I watched this YouTube video to understand the Isolation Forest algorithm : https://www.youtube.com/watch?v=kN--TRv1UDY
# I watched this YouTube video to help me implement the Isolation Forest algorithm : https://www.youtube.com/watch?v=5NcbVYb7v4Y
# and this video for OneClassSVM : https://www.youtube.com/watch?v=55l6keimE8M


def anomaly_detection_OneClassSVM(df_learning, nu=0.05) -> 'indexes':
    """
    This function detects anomalies in the data using the One-Class SVM algorithm
    :param df_learning: the data frame with the selected columns for learning
    :param nu: the proportion of anomalies desired
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
    :param contamination: influence the number of anomalies detected
    :return: the indexes of the anomalies
    """
    anomaly_detector = IsolationForest(bootstrap=bootstrap, contamination=contamination)
    result = anomaly_detector.fit_predict(df_learning)
    anomalies_indexes = np.where(result == -1)

    return anomalies_indexes


def plot_anomalies(df, df_learning, contamination=0.1):
    """
    This function plots the trees with all variables (6) by pairs.
    The color indicates which are the anomalies detected by the Isolation Forest algorithm.
    On the diagonal, you can see the distribution of anomalies in each variable.
    :param df_learning: the data frame with the selected columns for learning
    :param contamination: influence the number of anomalies detected
    :return:
    """
    model_IF = IsolationForest(contamination=contamination)
    model_IF.fit(df_learning)
    df_learning['anomaly'] = model_IF.predict(df_learning)
    palette = ['orange', 'blue']
    sns.pairplot(
        df_learning,
        vars=['haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_stadedev', 'fk_port'],
        hue='anomaly',
        palette=palette
    )
    plt.suptitle('Anomaly Detection Using Isolation Forest\n'
                 'number of anomalies: ' + str(len(df_learning[df_learning['anomaly'] == -1])),
                 y=0.95, fontsize=10)
    # plt.show()

    df['color'] = df_learning['anomaly'].apply(lambda x: 'red' if x == -1 else 'green')
    # Plot normal data points first
    plt.figure(figsize=(10, 5))
    plt.scatter(
        df[df['color'] == 'green']['longitude'],
        df[df['color'] == 'green']['latitude'],
        c='green',
        s=3
    )
    # Plot anomalies on top
    plt.scatter(
        df[df['color'] == 'red']['longitude'],
        df[df['color'] == 'red']['latitude'],
        c='red',
        s=3
    )
    plt.title('Anomaly Detection Using Isolation Forest\n'
              'number of anomalies: ' + str(len(df_learning[df_learning['anomaly'] == -1])))
    # plt.show()


def plot_number_anomalies(df_learning, ocsvm: bool = True, iforest: bool = True):
    """
    This function plots the number of anomalies detected by the OneClassSVM and IsolationForest algorithms
    for different values of nu and contamination, respectively.
    :param df_learning: the data frame with the selected columns for learning
    :param ocsvm: boolean : if False, skip the OneClassSVM algorithm and only plots results for IsolationForest
    :param iforest: same as ocsvm but for IsolationForest
    :return: plots
    """
    if ocsvm:
        nu_values = np.arange(0.01, 0.5, 0.01)

        # Lists to store the number of anomalies detected
        svm_anomalies = []

        # Detect anomalies using OneClassSVM for each nu value
        for nu in nu_values:
            anomalies_indexes = anomaly_detection_OneClassSVM(df_learning, nu)
            svm_anomalies.append(len(anomalies_indexes[0]))

        # Plot the number of anomalies detected for each nu value
        plt.figure(figsize=(10, 5))
        plt.plot(nu_values, svm_anomalies, marker='o')
        plt.title('Number of anomalies detected by OneClassSVM for different nu values')
        plt.xlabel('nu value')
        plt.ylabel('Number of anomalies')
        # plt.show()

    if iforest:
        contamination_values = np.arange(0.01, 0.5, 0.01)

        if_anomalies = []
        # Detect anomalies using IsolationForest for each contamination value
        for contamination in contamination_values:
            anomalies_indexes = anomaly_detection_IsolationForest(df_learning, contamination)
            if_anomalies.append(len(anomalies_indexes[0]))

        # Plot the number of anomalies detected for each contamination value
        plt.figure(figsize=(10, 5))
        plt.plot(contamination_values, if_anomalies, marker='o')
        plt.title('Number of anomalies detected by IsolationForest for different contamination values')
        plt.xlabel('contamination value')
        plt.ylabel('Number of anomalies')
        # plt.show()


def detect_anomalies(df_learning):
    anomalies_indexes = anomaly_detection_OneClassSVM(df_learning, nu=0.05)
    print(f'OneClassSVM detected {len(anomalies_indexes[0])} anomalies')

    anomalies_indexes_ = anomaly_detection_IsolationForest(df_learning, bootstrap=True, contamination=0.05)
    print(f'IsolationForest detected {len(anomalies_indexes_[0])} anomalies')
