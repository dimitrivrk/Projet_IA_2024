import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import requests
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
import folium
import sys
import math
import numpy as np
from PyQt5.QtGui import QIcon


tab_preci_moy = [45.57, 49.56, 44.1, 26.67, 46.47, 45.53, 42.63, 46.72, 36.15, 51.46, 49.68, 66.91]

tab_month = [
    [81.7, 79, 211, 38.6, 8.2, 59.4, 75, 26.6, 95.9, 37.8, 89.8, 96.6, 33, 26.6, 50.6, 33.6, 55.3, 29.4, 72.5, 51.2, 23.4, 63.8, 48.9, 71.1, 45.3, 95.6, 37.1, 35.1, 119.1, 60.4, 60.4],
    [19.9, 54.9, 95.1, 51.4, 96.2, 18.5, 61.1, 54.8, 43.2, 99.9, 7.4, 19.4, 46.2, 75.2, 101, 34.3, 36.6, 87, 27.6, 13.4, 39.1, 48.8, 51.2, 63.2, 35.7, 27.9, 62.6, 121.1, 39, 43.6, 6],
    [10.1, 89.6, 73.4, 36.5, 16.4, 38.8, 45.5, 47, 151.6, 55, 19.2, 41.2, 46.2, 63, 84.4, 85.6, 49.6, 42, 12.3, 35.3, 21.4, 23.4, 32.1, 84.6, 41.4, 61.5, 70.3, 52.9, 40.7, 13.4, 66.4],
    [63.1, 78, 30.8, 7.4, 12.3, 80.7, 69.6, 74, 99.6, 15.8, 49.4, 38, 70.2, 26.4, 2.8, 46.8, 43.6, 20.4, 27, 79.6, 20.8, 23.7, 33.6, 53.2, 4.6, 41.4, 29.2, 27.6, 27.1, 27.7, 63],
    [90.7, 175.2, 63.9, 111, 77, 18, 54.7, 77.1, 21.4, 44.2, 55.6, 35.4, 38, 116.2, 104.2, 53, 56.7, 27.5, 4.2, 28.6, 78.4, 107.3, 37.2, 132.6, 40.5, 54, 53.2, 21.4, 75.2, 31.9, 72],
    [35.5, 51.5, 33.6, 102.5, 132.3, 93.4, 52.1, 27, 70, 88.6, 141.4, 36.8, 20.4, 19.6, 100.7, 44.6, 3.8, 50.7, 76.1, 144.2, 75.7, 81.7, 28.6, 112, 25.7, 34.9, 45.6, 21.6, 148.7, 63.8, 33.5],
    [142.3, 39.9, 92.4, 22.7, 42.1, 54.8, 38.1, 123.2, 59.6, 75.6, 49.3, 60.8, 109, 59.4, 112.8, 58.8, 45, 61.8, 59.8, 92.4, 70, 62.8, 30.3, 28.4, 39.7, 77.8, 39.2, 21, 64.9, 11.5, 108.6],
    [37.8, 144.9, 77.8, 191.2, 64.4, 43.8, 136.4, 50.5, 64.4, 80.6, 29, 121.4, 60.4, 172.4, 70.1, 82.8, 34.6, 129.5, 81.8, 32.2, 47.6, 99.2, 59.2, 37, 67.6, 56.1, 51.6, 34.1, 48.3, 11.8, 55.5],
    [93.7, 93.9, 80.9, 22.7, 14.8, 72.1, 78.3, 56.4, 98.1, 19.9, 2.08, 29.6, 33.6, 49.2, 21.6, 76, 23.9, 53.2, 52, 19.9, 74.8, 22.4, 79, 38.6, 61.6, 49.2, 31.1, 58.5, 45.1, 86.6, 53.7],
    [114.2, 45.7, 8.9, 52.1, 71.2, 99.1, 51.7, 103.6, 63.4, 73.4, 31.8, 61.8, 41.2, 50.8, 67, 59, 68.6, 40.6, 33.5, 67, 110, 68.2, 45.3, 28.5, 34.7, 32, 98.5, 96.1, 82.7, 34.6, 87.8],
    [47.4, 56.7, 15.3, 137.2, 72.6, 98.3, 33.8, 91.4, 77.8, 128.7, 43.2, 20.4, 47, 53.4, 46.4, 40.4, 67.8, 78, 19.7, 37.9, 88.7, 48.4, 49.6, 17, 76, 47.6, 81.8, 28, 33.9, 54.1, 78.7],
    [178.3, 74.4, 61.2, 67.5, 54.3, 65.6, 150.4, 60.8, 34.9, 100.6, 44.8, 41.8, 31.2, 76.6, 55, 31.4, 66.8, 41.7, 131.9, 73.9, 71, 71.9, 28.7, 26, 82, 103.6, 95.9, 91.5, 63.2, 48.5, 40.6]
]

# Un dictionnaire qui en entrer prend un numero de mois et ressort son nombre de jours
nb_jours_par_mois = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

API_KEY = 'L89R4BL63KR39FPW4HKZVX5NS'

def get_weather_data_wind(date):
    """
    Récupère les données météorologiques liées au vent pour une date donnée.

    Args:
        date (str): La date pour laquelle les données météorologiques doivent être récupérées.

    Returns:
        dict: Un dictionnaire contenant les données météorologiques liées au vent, comprenant les clés suivantes :
            - 'temperature' : la température pour la date donnée.
            - 'humidity' : l'humidité pour la date donnée.
            - 'wind_speed' : la vitesse du vent maximale pour la date donnée.
            - 'wind_gust_speed' : la vitesse maximale des rafales de vent pour la date donnée. Si aucune donnée n'est disponible, la valeur de 'wind_speed' est utilisée.

    """
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/saint%20quentin/{date}/{date}?unitGroup=metric&elements=datetimeEpoch%2Ctemp%2Chumidity%2Cwindspeedmax%2Cwindgust&key={API_KEY}&contentType=json"
    response = requests.get(url)
    data = response.json()
    weather_data = {
        'temperature': data['days'][0]['temp'],
        'humidity': data['days'][0]['humidity'],
        'wind_speed': data['days'][0]['windspeedmax'],
        'wind_gust_speed': data['days'][0]['windgust']
    }
    if weather_data['wind_gust_speed'] is None:
        weather_data['wind_gust_speed'] = weather_data['wind_speed']
    
    return weather_data

def get_weather_data_drought(date, date2=None):
    """
    Récupère les données météorologiques liées à la sécheresse pour une période donnée.

    Args:
        date (str): La date de début de la période au format 'YYYY-MM-DD'.
        date2 (str, optional): La date de fin de la période au format 'YYYY-MM-DD'. Si non spécifiée, seule la date de début sera utilisée.

    Returns:
        dict: Un dictionnaire contenant les données météorologiques liées à la sécheresse. Les données incluent la somme des précipitations sur la période spécifiée.

    """
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/saint%20quentin/{date}/{date2}?unitGroup=metric&elements=precip&key={API_KEY}&contentType=json"
    response = requests.get(url)
    data = response.json()
    precipitation = [x['precip'] for x in data['days'] if x['precip'] is not None]
    weather_data = {
        'precip': sum(precipitation)
    }
    
    return weather_data

def Script_Client_3(date, date2=None, categorie='wind',path='data_test.json'):

    new_data = pd.read_json(path, lines=True)
    model_artifacts = joblib.load('model_artifacts.pkl')
    scaler = model_artifacts['scaler']
    encoder = model_artifacts['encoder']
    encoder_bool = model_artifacts['encoder_bool']
    clf_subset = model_artifacts['random_forest_model_subset']
    top_20_features = model_artifacts['top_20_features']

    selected_columns = ['longitude', 'latitude', 'haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim',
                        'fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'fk_revetement', 'villeca',
                        'feuillage', 'remarquable']

    numerical = ['longitude', 'latitude', 'haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim']
    categorical = ['fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'villeca', 'feuillage']
    categorical_bool = ['remarquable', 'fk_revetement']
    target = 'fk_arb_etat'

    new_data['fk_arb_etat'] = new_data['fk_arb_etat'].apply(lambda x: 1 if x == 'Essouché' else 0)

    X_new = new_data[selected_columns]
    y_new = new_data['fk_arb_etat']

    new_data_numerical = pd.DataFrame(scaler.transform(X_new[numerical]), columns=numerical)
    new_data_categorical = pd.DataFrame(encoder.transform(X_new[categorical]), columns=encoder.get_feature_names_out(categorical))
    new_data_bool = pd.DataFrame(encoder_bool.transform(X_new[categorical_bool]), columns=categorical_bool)
    new_data_encoded = pd.concat([new_data_numerical, new_data_categorical, new_data_bool], axis=1)
    new_data_subset = new_data_encoded[top_20_features]

    y_pred = clf_subset.predict(new_data_subset)
    y_pred_proba = clf_subset.predict_proba(new_data_subset)[:, 1]

    df_results = pd.DataFrame(new_data_subset, index=new_data.index)
    df_results['risque_deracinement'] = y_pred
    df_results['probabilite_risque'] = y_pred_proba
    
    conf_matrix = confusion_matrix(y_new, y_pred,normalize='true')
    accuracy = accuracy_score(y_new, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)

    df_results['latitude'] = new_data['latitude']
    df_results['longitude'] = new_data['longitude']
    
    condition = 0.5
    
    if categorie == 'wind':
        weather_data = get_weather_data_wind(date)
        if weather_data['wind_gust_speed'] >= 300:
            condition = 0.001
        elif weather_data['wind_gust_speed'] >= 250:
            condition = 0.01
        elif weather_data['wind_gust_speed'] >= 200:
            condition = 0.1
        elif weather_data['wind_gust_speed'] >= 150:
            condition = 0.2
        elif weather_data['wind_gust_speed'] >= 90:
            condition = 0.3
        elif weather_data['wind_gust_speed'] >= 75:
            condition = 0.4
        else:
            condition = 0.5
            
        df_results['temperature'] = weather_data['temperature']
        df_results['humidity'] = weather_data['humidity']
        df_results['wind_speed'] = weather_data['wind_speed']
        df_results['wind_gust_speed'] = weather_data['wind_gust_speed']
        

        map = folium.Map(location=[df_results['latitude'].mean(), df_results['longitude'].mean()], zoom_start=13)

        for _, row in df_results[df_results['probabilite_risque'] > condition].iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='red',
                fill=True,
                fill_color='red',
                popup=(
                    f"Arbre à risque:<br>"
                    f"Probabilité de risque: {row['probabilite_risque']:.2f}<br>"
                    f"Température: {row['temperature']}°C<br>"
                    f"Humidité: {row['humidity']}%<br>"
                    f"Vitesse du vent: {row['wind_gust_speed']} km/h"
                )
            ).add_to(map)

    elif categorie == 'drought':
        start_date = f"{date2}-01"
        month = int(date2.split('-')[1])
        end_date = f"{date2}-{nb_jours_par_mois[month]}"
        weather_data = get_weather_data_drought(start_date, end_date)
        month = int(date2.split('-')[1])
        moyenne_historique = np.mean(tab_month[month-1])
        ecat_type = np.std(tab_month[month-1])
        SPI = (weather_data['precip'] - moyenne_historique) / ecat_type
        df_results['SPI'] = SPI
        
        map = folium.Map(location=[df_results['latitude'].mean(), df_results['longitude'].mean()], zoom_start=13)
        print("SPI : ",SPI,"\n")
        message = ""
        if SPI >= 2.0:
            message = "Pluie extrêmement abondante"
            color = 'darkblue'
        if 1.5 <= SPI < 2.0:
            message = "Pluie très abondante"
            color = 'blue'
        elif 1.0 <= SPI < 1.5:
            message = "Pluie modérément abondante"
            color = 'lightblue'
        elif -1.0 < SPI < 1.0:
            message = "Conditions normales"
            color = 'green'
        elif -1.5 <= SPI < -1.0:
            message = "Secheresse modérée"
            color = 'yellow'
        elif -2.0 <= SPI < -1.5:
            message = "Sécheresse sévère"
            color = 'orange'
        elif SPI < -2.0:
            message = "Sécheresse extrême"
            color = 'red'
        
            
        # Ajouter tous les arbres à la carte
        for _, row in df_results.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='green',
                fill=True,
                fill_color='green',
                popup=(
                    f"Arbre à risque:<br>"
                    f"SPI: {row['SPI']:.2f}<br>"
                    f"Message: {message}"
                )
            ).add_to(map)
        
        # Ajouter une popup centrale avec le message
        folium.Marker(
            location=[df_results['latitude'].mean(), df_results['longitude'].mean()],
            popup=message,
            icon=folium.Icon(color=color)
        ).add_to(map)
            

    map.save("map.html")
    webbrowser.open("map.html")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Risque déracinement et stress hydrique")
        appIcon = QIcon("background.png")  # Remplacez par le chemin vers votre icône

        # Appliquer l'icône à la fenêtre principale
        self.setWindowIcon(appIcon)
        self.setGeometry(900, 600, 900, 400)
        self.setFixedWidth(self.width())


        self.wind_button = QPushButton("Wind", self)
        self.drought_button = QPushButton("Drought", self)
        self.date_input_label = QLabel("Date:")
        self.date_input = QLineEdit(self)
        self.date_input.setPlaceholderText("Entrez la date (YYYY-MM-DD)")
        
        self.month_label = QLabel("Mois:")
        self.month_selector = QComboBox(self)
        self.month_selector.addItems(["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"])
        self.year_label = QLabel("Année:")
        self.year_selector = QComboBox(self)
        self.year_selector.addItems([str(year) for year in range(2000, 2031)])
        
        self.load_button = QPushButton("Charger et Afficher la Carte", self)
        
        self.layout_buttons = QHBoxLayout()
        self.layout_buttons.addWidget(self.wind_button)
        self.layout_buttons.addWidget(self.drought_button)
        
        self.layout_inputs = QVBoxLayout()
        self.layout_inputs.addWidget(self.date_input_label)
        self.layout_inputs.addWidget(self.date_input)
        self.layout_inputs.addWidget(self.month_label)
        self.layout_inputs.addWidget(self.month_selector)
        self.layout_inputs.addWidget(self.year_label)
        self.layout_inputs.addWidget(self.year_selector)
        self.layout_inputs.addWidget(self.load_button)

        layout = QVBoxLayout()
        layout.addLayout(self.layout_buttons)
        layout.addLayout(self.layout_inputs)
        self.setLayout(layout)

        self.wind_button.clicked.connect(self.show_wind_input)
        self.drought_button.clicked.connect(self.show_drought_input)
        self.load_button.clicked.connect(self.load_map)

        self.adjustSize()

        self.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: black;
                border-radius: 10px;
                padding: 10px;
                border: 4px solid #ccc;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
            }
            
            QVBoxLayout {
                background: url('background.png');
            }
        """)

    def show_wind_input(self):
        self.clear_inputs()
        self.layout_inputs.addWidget(self.date_input_label)
        self.layout_inputs.addWidget(self.date_input)
        self.layout_inputs.addWidget(self.load_button)
        #self.adjustSize()

    def show_drought_input(self):
        self.clear_inputs()
        self.layout_inputs.addWidget(self.month_label)
        self.layout_inputs.addWidget(self.month_selector)
        self.layout_inputs.addWidget(self.year_label)
        self.layout_inputs.addWidget(self.year_selector)
        self.layout_inputs.addWidget(self.load_button)
        self.adjustSize()

    def clear_inputs(self):
        for i in reversed(range(self.layout_inputs.count())): 
            widget = self.layout_inputs.itemAt(i).widget()
            if widget:
                self.layout_inputs.removeWidget(widget)
                widget.setParent(None)

    def load_map(self):
        if self.date_input.isVisible():
            date = self.date_input.text()
            Script_Client_3(date, categorie='wind',path='data_test.json')
        elif self.month_selector.isVisible() and self.year_selector.isVisible():
            month = self.month_selector.currentIndex() + 1
            year = self.year_selector.currentText()
            month_year = f"{year}-{month:02d}"
            Script_Client_3(None, date2=month_year, categorie='drought',path='data_test.json')


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
