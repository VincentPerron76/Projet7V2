import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    """
    Fonction d'initialisation qui charge le modèle lors du démarrage de l'API.
    """
    global model
    # Chargement du modèle sauvegardé depuis Azure
    model_path = Model.get_model_path('model_P7')  # Remplacez 'votre_modele' par le nom du modèle
    model = joblib.load(model_path)

def run(input_data):
    """
    Fonction qui prend en entrée les données sous forme JSON et renvoie les prédictions du modèle.
    """
    try:
        # Parse les données d'entrée en un tableau numpy
        data = np.array(json.loads(input_data)["data"])
        
        # Faire des prédictions avec le modèle
        predictions = model.predict(data)
        
        # Retourner les résultats sous forme JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        # En cas d'erreur, retourner l'erreur sous forme JSON
        result = str(e)
        return json.dumps({"error": result})
