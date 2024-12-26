# Importer les bibliothèques nécessaires
from flask import Flask, request, jsonify
import joblib
import pandas as pd



# Charger le pipeline de production
pipeline = joblib.load("artifacts/production_pipeline.joblib")

# Charger les données des clients (exemple d'un fichier CSV)
client_data = pd.read_csv("artifacts/testclient.csv", index_col="SK_ID_CURR")


# Créer une instance Flask
app = Flask(__name__)

# Route pour vérifier que l'API fonctionne
@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de prédiction ! Utilisez /predict pour obtenir des résultats."


@app.route("/bonjour", methods=["GET"])
def bonjour():
    SK_ID_CURR = request.args.get('SK_ID_CURR')
    return f"vous avez demandé l'identifiant {SK_ID_CURR}"

# Route pour prédire à partir d'un identifiant client
@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Récupérer l'identifiant du client depuis les paramètres de l'URL
        SK_ID_CURR = request.args.get("SK_ID_CURR")  # Récupérer le paramètre SK_ID_CURR

        # Vérifier si l'identifiant est fourni
        if SK_ID_CURR is None:
            return jsonify({"error": "SK_ID_CURR est requis en tant que paramètre GET."}), 400

        # Vérifier si l'identifiant du client existe dans les données
        if int(SK_ID_CURR) not in client_data.index:
            return jsonify({"error": f"Client avec id {SK_ID_CURR} introuvable."}), 404

        # Extraire les données pour ce client
        input_data = client_data.loc[[int(SK_ID_CURR)]]

        # Faire la prédiction
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.30
        predictions = (probabilities >= seuil_personnalise).astype(int)

        # Retourner les résultats sous forme de JSON
        response = {
            "SK_ID_CURR": SK_ID_CURR,
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'application Flask sur le port 5001
if __name__ == "__main__":
    app.run(debug=True, port=5001)
