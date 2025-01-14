import joblib
import pandas as pd
import os


# Charger le pipeline de production avec un chemin relatif correct
pipeline_path = os.path.join("..", "artifacts", "production_pipeline.joblib")  # Remonter d'un niveau et accéder à 'artifacts'
pipeline = joblib.load(pipeline_path)

# Charger les données des clients à tester
client_data_path = os.path.join("..", "data", "test_client.csv")  # Remonter d'un niveau et accéder à 'data'
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