import joblib
import pandas as pd
import os

# Spécification des chemins absolus en dur
pipeline_path = "/opt/render/project/src/artifacts/production_pipeline.joblib"
print(f"Chargement du pipeline depuis : {pipeline_path}")
pipeline = joblib.load(pipeline_path)

# Spécification du chemin absolu pour les données des clients
client_data_path = "/opt/render/project/src/data/test_client.csv"
print(f"Chargement des données des clients depuis : {client_data_path}")
client_data = pd.read_csv(client_data_path, index_col="SK_ID_CURR")


client_ids_to_test = [241603, 350714, 211868, 268880, 305344, 180213, 398182, 443859, 259596,187836]

for client_id in client_ids_to_test:
    if client_id in client_data.index:
        input_data = client_data.loc[[client_id]]
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.14
        prediction = (probabilities >= seuil_personnalise).astype(int)[0]
        print(f"Client ID: {client_id}, Prédiction: {prediction}")
    else:
        print(f"Client ID: {client_id} non trouvé dans les données.")