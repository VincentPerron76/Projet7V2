
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


@app.route("/predict", methods=["GET"])
def predict():
    try:
        SK_ID_CURR = request.args.get("SK_ID_CURR")

        if not SK_ID_CURR or not SK_ID_CURR.isdigit():
            return jsonify({"error": "SK_ID_CURR doit être un entier valide."}), 400

        SK_ID_CURR = int(SK_ID_CURR)
        if SK_ID_CURR not in client_data.index:
            return jsonify({"error": f"Client avec id {SK_ID_CURR} introuvable."}), 404

        input_data = client_data.loc[[SK_ID_CURR]]

        preprocessed_data = pipeline.named_steps['imputer'].transform(input_data)
        preprocessed_data = pipeline.named_steps['scaler'].transform(preprocessed_data)

        probabilities = pipeline.predict_proba(preprocessed_data)[:, 1]
        seuil_personnalise = 0.14
        predictions = (probabilities >= seuil_personnalise).astype(int)

        model = pipeline['classifier']
        explainer = shap.TreeExplainer(model)
        base_value = explainer.expected_value

        shap_values = explainer.shap_values(preprocessed_data)

        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_to_use = shap_values

        feature_names = input_data.columns.tolist()
        feature_values = input_data.iloc[0].to_dict()

        shap_contributions = {
            feature: {
                "shap_value": shap_value,
                "original_value": feature_values[feature]
            }
            for feature, shap_value in zip(feature_names, shap_values_to_use[0])
        }

        shap_sum = np.sum(shap_values_to_use[0])
        shap_proba = 1 / (1 + np.exp(-(base_value + shap_sum)))
        logit_at_threshold = -np.log(1 / seuil_personnalise - 1)

        response = {
            "SK_ID_CURR": SK_ID_CURR,
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0]),
            "shap_contributions": shap_contributions,
            "base_value": float(base_value),
            "shap_pivot": float(logit_at_threshold),
            "total_shap": shap_sum,
            "shap_proba": float(shap_proba)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', debug=True, port=port)  # Ajout de port=port ici pour utiliser la variable port

 