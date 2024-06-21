import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.feature_selection import f_classif
from predictionAge import load_and_preprocess_data
matplotlib.use('Qt5Agg')


def correlations_plot():
    datarbre = load_and_preprocess_data(split=False)
    datarbre = datarbre[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'clc_nbr_diag', 'feuillage_Conifère', 'feuillage_Feuillu', 'age_estim']]
    corr_age_estim = datarbre.corr()['age_estim']

    sns.barplot(x=corr_age_estim.index, y=corr_age_estim.values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Corrélation des variables quantitatives avec l'âge estimé")

    data = pd.read_csv('Data_Arbre.csv')
    # select only the columns that are categorical
    data = data[['age_estim', 'clc_quartier', 'fk_port', 'fk_nomtech', 'remarquable']]

    # encode them and split them, else there are too many categories to plot
    data_quartier = pd.get_dummies(data, columns=['clc_quartier']).drop(columns=['fk_port', 'fk_nomtech', 'remarquable'])
    corr_age_estim = data_quartier.corr()['age_estim']
    plt.figure()
    sns.barplot(x=corr_age_estim.index, y=corr_age_estim.values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Corrélation des quartiers avec l'âge estimé")

    data_port = pd.get_dummies(data, columns=['fk_port']).drop(columns=['clc_quartier', 'fk_nomtech', 'remarquable'])
    corr_age_estim = data_port.corr()['age_estim']
    plt.figure()
    sns.barplot(x=corr_age_estim.index, y=corr_age_estim.values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Corrélation de fk_port avec l'âge estimé")

    data_nomtech = pd.get_dummies(data, columns=['fk_nomtech']).drop(columns=['fk_port', 'clc_quartier', 'remarquable'])
    corr_age_estim = data_nomtech.corr()['age_estim']
    plt.figure()
    sns.barplot(x=corr_age_estim.index, y=corr_age_estim.values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Corrélation de fk_nomtech avec l'âge estimé")

    data_remarquable = pd.get_dummies(data, columns=['remarquable']).drop(columns=['fk_port', 'fk_nomtech', 'clc_quartier'])
    corr_age_estim = data_remarquable.corr()['age_estim']
    plt.figure()
    sns.barplot(x=corr_age_estim.index, y=corr_age_estim.values)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Corrélation de remarquable avec l'âge estimé")


def anova_plot():
    """
    perform anova test with categorical variables and plot the results
    :return:
    """
    data = pd.read_csv('Data_Arbre.csv')
    # select only the columns that are categorical
    data = data[['age_estim', 'clc_quartier', 'fk_port', 'fk_nomtech', 'remarquable']]

    mean_p_values = {}
    for variable in ['clc_quartier', 'fk_port', 'fk_nomtech', 'remarquable']:
        data_encoded = pd.get_dummies(data[[variable, 'age_estim']])

        X = data_encoded.drop(columns=['age_estim'])
        y = data_encoded['age_estim']

        _, p_values = f_classif(X, y)
        mean_p_values[variable] = np.mean(p_values)

    # Plot mean p-values
    plt.figure()
    sns.barplot(x=list(mean_p_values.keys()), y=list(mean_p_values.values()))
    plt.title("Mean p-values of categorical variables with the estimated age")


if __name__ == '__main__':
    correlations_plot()

    anova_plot()

    plt.show()
