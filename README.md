# Projet_IA_2024




Pour le Besoin Client 3 :

Il vous faut vous rendre dans le dossier `feature3` et exécuter le fichier `Clien_3.ipynb`. Ce fichier sert à créer et entraîner le modèle, puis à les encoder et les sauvegarder en format .plk.

Si vous souhaitez exécuter le script du besoin 3, lancez la commande "python Script3.py" dans le terminal en vous assurant d'être dans le dossier `feature3` à ce moment-là.

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