# Projet Prédictif : Prêt Bancaire

Ce projet a pour objectif de prédire l'acceptation ou le refus d'un prêt bancaire en utilisant un modèle de machine learning. Le modèle a été conçu en tenant compte des spécificités des données et en traitant un déséquilibre de classes. Le projet a été déployé via Flask et est accessible à travers une API.

## Répertoires et fichiers

### Répertoire `Notebook`
Ce répertoire contient le fichier **`projet7v6.ipynb`** qui comprend :
- L'exploration des données,
- La recherche et la sélection du modèle prédictif,
- La validation des hyperparamètres et des features utilisés pour le modèle final.

### Répertoire `Models`
Ce répertoire contient les éléments suivants :
- **`best_model_params.json`** : Sauvegarde des meilleurs hyperparamètres du modèle retenu issus de l'exploration réalisée dans le fichier `projet7v6.ipynb`.
- **`selected_features.csv`** : Liste des features sélectionnées utilisées dans le modèle retenu.
- **`train_pipeline.ipynb`** : Notebook contenant le pipeline de préproduction avec les étapes de préparation des données (scaling, imputation).
- **`production_pipeline.py`** : Script Python du pipeline de production, incluant les étapes nécessaires pour préparer les données en production et effectuer les prédictions.

### Répertoire `Artifacts`
Ce répertoire contient les sauvegardes des objets essentiels pour la production du modèle, notamment :
- **`imputer.joblib`** : Sauvegarde du modèle d'imputation pour gérer les valeurs manquantes.
- **`lgmb_model.joblib`** : Sauvegarde du modèle LightGBM entraîné.
- **`scaler.joblib`** : Sauvegarde de l'objet de scaling pour normaliser les données.
- **`preprocessing_pipeline.joblib`** : Sauvegarde du pipeline de prétraitement des données (scaling, imputation).
- **`production_pipeline.joblib`** : Sauvegarde du pipeline de production pour la prédiction.

### Répertoire `Script`
Ce répertoire contient les scripts nécessaires pour l'API et les tests unitaires :
- **`api.py`** : Script pour exposer l'API permettant d'effectuer des prédictions à partir du modèle.
- **`stream.py`** : Script pour intégrer l'API dans une interface utilisateur avec Streamlit.
- **`api_pytest.py`** : Script pour tester l'API avec **pytest**.
- **`trouve_clients_test_uni.py`** : Script pour trouver des clients à utiliser dans les tests unitaires.
- **`test_predict_valid.py`** : Script pour valider les prédictions du modèle en production à l'aide de **pytest**.

### Répertoire `Data`
Ce répertoire contient des données utilisées pour la démonstration et les tests unitaires :
- **`test_client.csv`** : Fichier contenant un nombre restreint de clients, utilisé pour la démonstration et les tests unitaires.

### Fichier `requirementsV2.txt`
Ce fichier contient la liste des bibliothèques nécessaires pour faire fonctionner le projet

### Répertoire `Drift`
Ce répertoire contient la page HTML des résultats de l'analyse de drift des données :
- **`data_drift.html`** : Page HTML générée par l'analyse de drift, indiquant les variables dont le comportement a évolué au fil du temps.
- **`train_pipeline.ipynb`**: Notebook contenant le code pour l'analyse du drift

## Installation

1. Clonez ce dépôt sur votre machine locale :
   ```bash
   git clone https://github.com/VincentPerron76/Projet7V2
