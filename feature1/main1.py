import matplotlib.pyplot as plt

from size_classification import *
from anomaly_detection import *


def main1():
    """
    This function runs 3 steps:
    1- plots the silhouette scores, inertia and explained variance to choose the best number of clusters (k)
    2- show the trees clustered on maps, you first need to enter the number of clusters
    3- detect and plot anomalies
    :return:
    """
    df, df_learning = load_data()

    print('''
        This function runs 3 steps:
        1- plots the silhouette scores, inertia and explained variance to choose the best number of clusters (k)
           close the plots to go to step 2
        2- show the trees clustered on maps, you first need to enter the number of clusters
           close the plot to go to step 3
        3- detect and plot anomalies, the data is plotted by pair of varaibles and 
           on the diagonal you can see the distribution of anomalies in each variable
    \n''')

    print('plotting ...')
    evaluate_classifications(df_learning)
    plt.show()
    k = int(input('Choose a number of clusters: '))

    df = cluster(df, df_learning, k)
    print('plotting ...')
    plot_clusters(df, k)
    folium_map(df, k)
    print('\nThe folium/leaflet map is saved in the file map.html\n')
    plt.show()

    detect_anomalies(df_learning)
    print('plotting ...')
    plot_anomalies(df_learning, contamination=0.1)
    plot_number_anomalies(df_learning, ocsvm=True, iforest=True)
    plt.show()


if __name__ == '__main__':
    main1()
