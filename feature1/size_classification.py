import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import folium
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from anomaly_detection import anomaly_detection_OneClassSVM, anomaly_detection_IsolationForest, plot_anomalies


def load_data() -> 'dataframes':
    """
    This function loads the data from the csv file and prepares it for learning
    :return: the data frame with all columns and the data frame with the selected columns for learning
    """
    df = pd.read_csv('Data_Arbre.csv')

    # choose the columns to keep
    df_learning = df[['haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_stadedev', 'fk_port']].copy()

    # use OrdinalEncoder to encode the categorical columns : fk_stadedev and fk_port
    encoder = OrdinalEncoder()
    df_learning[['fk_stadedev', 'fk_port']] = encoder.fit_transform(df_learning[['fk_stadedev', 'fk_port']])

    return df, df_learning


# I chose K-means because you can choose the number of clusters, and it is easy to understand :)
# I watched this YouTube video to undersdand how K-means works : https://www.youtube.com/watch?v=4b5d3muPQmA
def cluster(df, df_learning, k=3) -> 'dataframe':
    """
    This function clusters the trees by their size using the KMeans algorithm. The cluster n° is stored in df
    :param df: the original data frame with all columns
    :param df_learning: the data frame with the selected columns for learning
    :param k: the number of clusters
    :return: the original data frame with a new column 'cluster' that indicates the cluster n°
    """
    # clustering (apprentissage non supervisé)
    kmeans = KMeans(n_clusters=k)
    df['cluster'] = kmeans.fit_predict(df_learning)

    # metrics
    silhouette = silhouette_score(df_learning, df['cluster'])
    print(f'Silhouette score : {silhouette}')
    print(f'Inertia : {kmeans.inertia_}')

    return df


def evaluate_classifications(df_learning, max_clusters: int = 10) -> 'plots':
    """
    This function evaluates the classifications for different number of clusters (k)
    by plotting the silhouette scores, inertia and explained variance.
    The goal is to find the best number of clusters.
    With the inertia method there is no clear elbow,
    and with the silhouette score the best k is 2 but the curve is monotonic
    :param df_learning: the data frame with the selected columns for learning
    :param max_clusters: choose the range of k (number of clusters) to evaluate
    :return: check the plots
    """
    # Prepare an empty list to store the silhouette scores, inertia and explained variance
    silhouette_scores = []
    inertia_list = []
    explained_variance = []

    # Calculate total variance
    total_variance = np.var(df_learning, axis=0).sum()

    # Iterate over a range of cluster numbers
    for i in range(2, max_clusters+1):
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_learning)

        # Calculate silhouette score and inertia for the current number of clusters
        silhouette = silhouette_score(df_learning, kmeans.labels_)
        inertia = kmeans.inertia_

        # Calculate explained variance
        explained_var = (total_variance - inertia) / total_variance

        # Append the scores to the respective lists
        silhouette_scores.append(silhouette)
        inertia_list.append(inertia)
        explained_variance.append(explained_var)

    # Plot Silhouette scores
    plt.figure(num=1, figsize=(10,5))
    plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
    plt.title('Silhouette scores for different numbers of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    # plt.show()

    # Plot inertia
    plt.figure(num=2, figsize=(10,5))
    plt.plot(range(2, max_clusters+1), inertia_list, marker='o')
    plt.title('Inertia for different numbers of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    # plt.show()

    # Plot explained variance
    plt.figure(num=3, figsize=(10,5))
    plt.plot(range(2, max_clusters+1), explained_variance, marker='o')
    plt.title('Explained variance for different numbers of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Explained variance')
    # plt.show()


def plot_clusters(df, k=3) -> 'plot map':
    """
    This function plots the trees on a basic 2D space with different colors for each cluster
    :param df: the original data frame (with the new column 'cluster')
    :param k: number of clusters
    :return: check the plots
    """
    colors = colors = cm.rainbow(np.linspace(0, 1, k))
    labels = [f'Cluster {i+1}' for i in range(k)]

    for i in range(k):
        plt.scatter(df[df['cluster'] == i]['longitude'], df[df['cluster'] == i]['latitude'],
                    s=0.5,
                    color=colors[i],
                    label=labels[i])

    # increase the size of dots in the legend, they were too small
    [handle.set_sizes([10]) for handle in plt.legend(loc='best').legend_handles]
    plt.title('basic map')
    # plt.show()


def folium_map(df, k=3) -> 'map in html file':
    """
    This function store a folium/leaflet map in a html file.
    The trees are colored by their cluster and
    if k<=3, you can click on a dot to see if it is a small, medium or big tree
    :param df: the original data frame (with the new column 'cluster')
    :param k:
    :return:
    """
    m = folium.Map(location=[49.84050020512298, 3.2932636093638927], zoom_start=12)
    colors = [mcolors.rgb2hex(c) for c in cm.rainbow(np.linspace(0, 1, k))]
    df['color'] = df['cluster'].apply(lambda x: colors[x])

    if k <=3:
        clusters_avg_dimensions = df.groupby('cluster')[['haut_tot', 'tronc_diam']].mean()
        clusters_avg_size = clusters_avg_dimensions['haut_tot']*clusters_avg_dimensions['tronc_diam']
        clusters_sorted = sorted(enumerate(clusters_avg_size), key=lambda x: x[1])
        clusters_sorted = [x[0] for x in clusters_sorted]
        if k == 2:
            size = ['small', 'big']
        if k == 3:
            size = ['small', 'medium', 'big']

    for i in range(0, len(df)):
        tree = df.iloc[i]
        folium.Circle(
            location=[tree['latitude'], tree['longitude']],
            radius=tree['haut_tot']*tree['tronc_diam']/2000,
            color=tree['color'],
            fill=True,
            fill_color=tree['color'],
            fill_opacity=0.7,
            popup=size[clusters_sorted.index(tree['cluster'])] if k<= 3 else f'Cluster {tree["cluster"]}'
        ).add_to(m)

    m.save('map.html')


def choose_k() -> 'int, plot map':
    """
    the user choose a number of clusters
    :return: display a basic map, tree color indicate the cluster
    """
    k = int(input('Enter the number of clusters : '))
    df, df_learning = load_data()
    df = cluster(df, df_learning, k)
    plot_clusters(df, k)
    return k


def plot_number_anomalies(ocsvm: bool = True, iforest: bool = True):
    df, df_learning = load_data()

    if ocsvm:
        nu_values = np.arange(0.01, 0.11, 0.01)

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


def plot_anomalies_():
    df, df_learning = load_data()
    plot_anomalies(df_learning)
#

if __name__ == '__main__':
    # find_best_k()
    # k = choose_k()
    # show_map(k)

    plot_anomalies_()
