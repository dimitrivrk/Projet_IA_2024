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
        3- detect and plot anomalies, the data is plotted by pair of variables and 
           on the diagonal you can see the distribution of anomalies in each variable
           (this step is quite long)
    \n''')

    print('plotting ...\n')
    evaluate_classifications(df_learning)
    plt.show()

    k = int(input('\033[34mChoose a number of clusters: \033[0m'))
    df = cluster(df, df_learning, k)
    print('plotting ...')
    plot_clusters(df, k)
    folium_map(df, k)
    print('\nThe folium/leaflet map is saved in the file map.html\n')
    plt.show()

    anom_IF, anom_OCSVM, anom_both = detect_anomalies(df_learning, nu=0.1)
    print('plotting ...')
    plot_anomalies(df, anom_IF, anom_OCSVM, anom_both)
    # plot_number_anomalies(df_learning, ocsvm=True, iforest=True)
    plt.show()


if __name__ == '__main__':
    main1()
