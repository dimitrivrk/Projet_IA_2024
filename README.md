# Projet_IA_2024

## Pour le besoin client 1 :
Vous pouvez éxecuter deux scripts, un pour le clustering (script_clustering.py) par la taille et un pour la détection d'anomalies (script_anomalies.py).  
script_clustering.py a besoin d'un paramètre : le nombre de clusters.  
script_anomalies.py a besoin d'un paramètre : nu (la proportion d'anomalies).  
Vous pouvez aussi lancer le fichier main1.py qui lance toutes les fonctionnalités, il vous demande de choisir le nombre de clusters et lance une détection d'anomalies avec nu = 0.1.

bibliothèques nécessaires :  
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import folium
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import seaborn as sns
```

## Pour le besoin client 2 :
Vous pouvez exécuter le script `script_predict_age.py` qui a besoin d'un paramètre : le nom du fichier JSON. Un fichier pickle est déjà présent avec les modèles entrainés.  
Vous pouvez aussi lancer le fichier `predictionAge.py` qui donne les métriques des 4 modèles.  
Le fichier `etude_variable.py` étudie la corrélation entre les variables et l'âge et affiche les p-values. Ses plots servent à justifier le choix des variables explicatives.  
Le fichier `grid_search.py` fait une recherche des meilleurs hyperparamètres pour les 4 modèles. Il est très long à s'exécuter.


bibliothèques nécessaires :  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
import joblib

```

## Pour le Besoin Client 3 :

Il vous faut vous rendre dans le dossier `Besoin_Client_3` et exécuter le fichier `Clien_3.ipynb`. Ce fichier sert à créer et entraîner le modèle, puis à les encoder et les sauvegarder en format .plk.

Si vous souhaitez exécuter le script du besoin 3, lancez la commande "python Script3.py" dans le terminal en vous assurant d'être dans le dossier `Besoin_Client_3` à ce moment-là.

Pour que cela fonctionne, vous devez importer les bibliothèques suivantes :

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from PyQt5.QtGui import QIcon
import folium
import requests
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
```

Assurez-vous d'installer ces bibliothèques si vous ne les avez pas déjà fait.