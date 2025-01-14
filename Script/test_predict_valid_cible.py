import joblib
import pandas as pd
import os

# Afficher le répertoire de travail actuel pour le débogage
print("Répertoire actuel :", os.getcwd())

# Définir le répertoire de base
base_dir = os.path.dirname(os.path.abspath(__file__))


# Utiliser un chemin relatif pour accéder à 'artifacts' (remonter d'un niveau depuis 'Script')
pipeline_path = os.path.join(base_dir, "..", "artifacts", "production_pipeline.joblib")
pipeline = joblib.load(pipeline_path)

# Charger les données des clients avec un chemin absolu
client_data_path = os.path.join(base_dir,"..", "data", "test_client.csv")
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