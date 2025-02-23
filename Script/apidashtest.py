from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import shap
import numpy as np

# Définition des chemins
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "..", "artifacts", "production_pipeline.joblib")
client_data_path = os.path.join(base_dir, "..", "data", "test_client.csv")

# Chargement du modèle et des données clients
pipeline = joblib.load(pipeline_path)
client_data = pd.read_csv(client_data_path, index_col="SK_ID_CURR")

# Calculer les statistiques sur l'ensemble du jeu de données
stats = client_data.describe().T[['mean', '25%', '50%', '75%', 'min', 'max']]
stats.rename(columns={'50%': 'median'}, inplace=True)

# Initialisation de l'API Flask
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

        # Extraction et prétraitement des données
        input_data = client_data.loc[[SK_ID_CURR]]
        preprocessed_data = pipeline.named_steps['imputer'].transform(input_data)
        preprocessed_data = pipeline.named_steps['scaler'].transform(preprocessed_data)

        # Prédiction
        probabilities = pipeline.named_steps['classifier'].predict_proba(preprocessed_data)[:, 1]
        seuil_personnalise = 0.14
        predictions = (probabilities >= seuil_personnalise).astype(int)

        # Explication SHAP
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

        # Création d'un DataFrame SHAP
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values_to_use[0],
            "original_value": [feature_values[f] for f in feature_names]
        })

        df_shap_value=shap_df.to_dict(orient="records")
      

        # Sélection des 5 contributeurs les plus positifs et négatifs
        top_positive = shap_df.nlargest(7, "shap_value").to_dict(orient="records")
        top_negative = shap_df.nsmallest(7, "shap_value").to_dict(orient="records")

        # Comparaison avec les statistiques des percentiles et des moyennes
        feature_comparison_pos = []
        for feature in top_positive:
            feature_name = feature['feature']
            client_value = feature['original_value']
            median_value = stats.loc[feature_name, 'median']
            perc_25 = stats.loc[feature_name, '25%']
            perc_75 = stats.loc[feature_name, '75%']
            

            feature_comparison_pos.append({
                "feature": feature_name,
                "client_value": client_value,
                "median": median_value,
                "25%": perc_25,
                "75%": perc_75,
                "shap_value": feature["shap_value"] 
            })

          # Comparaison avec les statistiques des percentiles et des moyennes
        feature_comparison_neg = []
        for feature in top_negative:
            feature_name = feature['feature']
            client_value = feature['original_value']
            median_value = stats.loc[feature_name, 'median']
            perc_25 = stats.loc[feature_name, '25%']
            perc_75 = stats.loc[feature_name, '75%']

            feature_comparison_neg.append({
                "feature": feature_name,
                "client_value": client_value,
                "median": median_value,
                "25%": perc_25,
                "75%": perc_75,
                "shap_value": feature["shap_value"] 
            })  

        

        # Construction de la réponse JSON
        response = {
            "SK_ID_CURR": SK_ID_CURR,
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0]),
            "base_value": float(base_value),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "feature_comparison_positive": feature_comparison_pos,
            "feature_comparison_negative": feature_comparison_neg,
            "tot_shape_value": float(np.sum(shap_values)),
            "fair_value": float(base_value) + float(np.sum(shap_values)),
            "seuil_fair_value": float (np.log(seuil_personnalise / (1 - seuil_personnalise))),
            "detail_des_shap_value": df_shap_value
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data_statistics", methods=["GET"])
def data_statistics():
    try:
        return jsonify(stats.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/dataset", methods=["GET"])
def get_dataset():
    df_data = client_data.copy()  # Crée une copie du DataFrame original
    df_data['SK_ID_CURR'] = df_data.index  # Ajoute explicitement l'index 'SK_ID_CURR' comme une colonne
    df_data = df_data.reset_index(drop=True)  # Réinitialise l'index sans l'inclure dans les colonnes
    df_data = df_data.to_dict(orient="records")  # Convertit en une liste de dictionnaires
    return jsonify(df_data)





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', debug=True, port=port)

 