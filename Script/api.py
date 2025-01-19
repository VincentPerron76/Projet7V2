from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Afficher le répertoire de travail actuel pour le débogage
print("Répertoire actuel :", os.getcwd())

# Définir le répertoire de base
base_dir = os.path.dirname(os.path.abspath(__file__))

# Utiliser un chemin relatif pour accéder à 'artifacts' (remonter d'un niveau depuis 'Script')
pipeline_path = os.path.join(base_dir, "..", "artifacts", "production_pipeline.joblib")
print(f"Chargement du pipeline depuis : {pipeline_path}")
pipeline = joblib.load(pipeline_path)

# Utiliser un chemin relatif pour accéder à 'data' (remonter d'un niveau depuis 'Script')
client_data_path = os.path.join(base_dir, "..", "data", "test_client.csv")
print(f"Chargement des données des clients depuis : {client_data_path}")
client_data = pd.read_csv(client_data_path, index_col="SK_ID_CURR")

# Creer une instance Flask
app = Flask(__name__)

# Route pour verifier que l'API fonctionne
@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de prediction ! Utilisez /predict pour obtenir des resultats."

@app.route("/bonjour", methods=["GET"])
def bonjour():
    SK_ID_CURR = request.args.get('SK_ID_CURR')
    return f"vous avez demande l'identifiant {SK_ID_CURR}"

# Route pour predire a partir d'un identifiant client
@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Recuperer l'identifiant du client depuis les parametres de l'URL
        SK_ID_CURR = request.args.get("SK_ID_CURR")  # Recuperer le parametre SK_ID_CURR

        # Verifier si l'identifiant est fourni
        if SK_ID_CURR is None:
            return jsonify({"error": "SK_ID_CURR est requis en tant que parametre GET."}), 400

        # Verifier si l'identifiant du client existe dans les donnees
        if int(SK_ID_CURR) not in client_data.index:
            return jsonify({"error": f"Client avec id {SK_ID_CURR} introuvable."}), 404

        # Extraire les donnees pour ce client
        input_data = client_data.loc[[int(SK_ID_CURR)]]

        # Faire la prediction
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.14
        predictions = (probabilities >= seuil_personnalise).astype(int)

        # Retourner les resultats sous forme de JSON
        response = {
            "SK_ID_CURR": SK_ID_CURR,
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', debug=True, port=port) # Ajout de port=port ici pour utiliser la variable port