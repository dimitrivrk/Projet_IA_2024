from size_classification import *
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You must enter the number of clusters as a parameter, it must be a positive integer')

    k = int(sys.argv[1])
    if k < 1:
        raise ValueError('You must enter the number of clusters as a parameter, it must be a positive integer')

    df, df_learning = load_data()
    df = cluster(df, df_learning, k)
    print('plotting ...')
    plot_clusters(df, k)
    folium_map(df, k)
    print('\nThe folium/leaflet map is saved in the file map.html\n')
    plt.show()
