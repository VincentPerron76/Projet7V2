from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import shap
import numpy as np

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

# Créer une instance Flask
app = Flask(__name__)

# Route pour vérifier que l'API fonctionne
@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de prediction ! Utilisez /predict pour obtenir des résultats."

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
            return jsonify({"error": "SK_ID_CURR est requis en tant que parametre GET."}), 400

        # Vérifier si l'identifiant du client existe dans les données
        if int(SK_ID_CURR) not in client_data.index:
            return jsonify({"error": f"Client avec id {SK_ID_CURR} introuvable."}), 404

        # Extraire les données pour ce client
        input_data = client_data.loc[[int(SK_ID_CURR)]]

        # Extraire les noms des features (étiquettes)
        feature_names = input_data.columns.tolist()

        # Faire la prédiction
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.14
        predictions = (probabilities >= seuil_personnalise).astype(int)

        # Extraire le modèle à partir du pipeline
        model = pipeline['classifier']

        # Créer l'explainer SHAP pour le modèle
        explainer = shap.TreeExplainer(model)

        # Appliquer l'imputation et le scaling avec le pipeline sur le client avant calcul SHAP
        preprocessed_data = pipeline.named_steps['imputer'].transform(input_data)
        preprocessed_data = pipeline.named_steps['scaler'].transform(preprocessed_data)

        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(preprocessed_data)

        
         
        #Base value du modèle global 
        base_value = explainer.expected_value[0]  # La base value du modèle global

        

        # Calcul de la somme des valeurs SHAP
        shap_values_sum = np.sum(shap_values[0])

        # Probabilité associée à la base value du modèle
        # application de la fonction sigmoïde à la base value pour obtenir la probabilité correspondante
        base_value_prob = 1 / (1 + np.exp(-base_value))  # Fonction sigmoïde

        # Calculer la logit correspondant à ce seuil (avant la fonction sigmoïde)
        logit_at_threshold = -np.log(1 / seuil_personnalise - 1)

    

        # Créer une liste de dictionnaires pour les valeurs SHAP et leurs étiquettes
        shap_result = [{"feature": feature_names[i], 
                        "shap_value": float(shap_values[1][i]),  # Convertir en float pour la sérialisation JSON
                        "initial_value": float(input_data[feature_names[i]].iloc[0])}  # Convertir en float
                       for i in range(len(feature_names))]

        # Retourner les résultats sous forme de JSON
        response = {
            "SK_ID_CURR": SK_ID_CURR,
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0]),
            "base_value":float(base_value),
            "shap_values_sum": float(shap_values_sum),
            "total_shap":float(base_value + shap_values_sum),
            "shap_pivot":float(logit_at_threshold),
            "shap_values": float(shap_result)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', debug=True, port=port)  # Ajout de port=port ici pour utiliser la variable port
