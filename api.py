# Importer les bibliothèques nécessaires
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Charger le pipeline de production
pipeline = joblib.load("artifacts/production_pipeline.joblib")

# Créer une instance Flask
app = Flask(__name__)

# Route pour vérifier que l'API fonctionne
@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de prédiction ! Utilisez /predict pour obtenir des résultats."

# Route pour effectuer une prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données envoyées par le client
        data = request.get_json()

        # Convertir les données JSON en DataFrame
        input_data = pd.DataFrame(data)

        # Faire les prédictions avec le pipeline
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.35
        predictions = (probabilities >= seuil_personnalise).astype(int)

        # Retourner les résultats sous forme de JSON
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Lancer l'application Flask sur le port 5001
if __name__ == "__main__":
    app.run(debug=True, port=5001)
