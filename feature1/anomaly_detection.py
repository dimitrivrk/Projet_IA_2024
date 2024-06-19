from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import numpy as np
import seaborn as sns


# Todo : try DBSCAN, Local Outlier Factor, Elliptic Envelope
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

    return anomalies_indexes[0]


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

    return anomalies_indexes[0]


def plot_anomalies(df, anom_IF, anom_OCSVM, anom_both):
    """
    This function plots the trees with pairs of variables.
    The fk_port varaible can't be shown, it would add too many subplots (one for each category).
    The color indicates which are the anomalies detected by the Isolation Forest and One Class SVM algorithms.
    On the diagonal, you can see the distribution of anomalies in each variable.
    """

    # add a column to the data frame to store the color of the anomalies. 0: normal, 1: both, 2: OCSVM, 3: IF
    df['color'] = 0
    df.loc[anom_OCSVM, 'color'] = 2
    df.loc[anom_IF, 'color'] = 3
    df.loc[anom_both, 'color'] = 1

    palette = ['yellow', 'orange', 'red']
    colors = ['blue', 'red', 'orange', 'yellow']
    # plot anomalies variables
    sns.pairplot(
        df,
        vars=['haut_tot', 'haut_tronc', 'tronc_diam'],
        hue='color',
        palette=colors
    )
    title = f"Anomaly Detection Using Isolation Forest and OCSVM\nnumber of anomalies: {len(anom_IF)}, {len(anom_OCSVM)}"
    plt.suptitle(title, y=0.95, fontsize=10)

    # plot anomalies on a map
    # Plot normal data points first
    plt.figure(figsize=(10, 5))
    plt.scatter(
        df[df['color'] == 0]['longitude'],
        df[df['color'] == 0]['latitude'],
        c='blue',
        s=3
    )
    # Plot anomalies on top
    plot_colors_anom = [colors[i] for i in df[df['color'] != 0]['color']]
    plt.scatter(
        df[df['color'] != 0]['longitude'],
        df[df['color'] != 0]['latitude'],
        c=plot_colors_anom,
        s=3
    )
    plt.title('Anomaly Detection Using Isolation Forest and OCSVM\n'
              f'number of anomalies detected by both methods: {len(anom_both)}')


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


def detect_anomalies(df_learning, nu=0.05, bootstrap=True):
    anomalies_OCSVM = anomaly_detection_OneClassSVM(df_learning, nu=nu)
    print(f'OneClassSVM detected {len(anomalies_OCSVM)} anomalies')

    anomalies_IF = anomaly_detection_IsolationForest(df_learning, bootstrap=bootstrap, contamination=nu)
    print(f'IsolationForest detected {len(anomalies_IF)} anomalies')

    anomalies_in_both = [anomaly_index for anomaly_index in anomalies_IF if anomaly_index in anomalies_OCSVM]
    print(f'anomalies detected by both methods ({len(anomalies_in_both)}):\n', anomalies_in_both)

    return anomalies_IF, anomalies_OCSVM, anomalies_in_both
