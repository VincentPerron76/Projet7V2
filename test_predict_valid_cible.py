import joblib
import pandas as pd

pipeline = joblib.load("artifacts/production_pipeline.joblib")
client_data = pd.read_csv("artifacts/testclient.csv", index_col="SK_ID_CURR")

client_ids_to_test = [234285,  235876,  130522,  112770, 249059, 151142,  280185,  136865, 372541,447393]

for client_id in client_ids_to_test:
    if client_id in client_data.index:
        input_data = client_data.loc[[client_id]]
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.20
        prediction = (probabilities >= seuil_personnalise).astype(int)[0]
        print(f"Client ID: {client_id}, Prédiction: {prediction}")
    else:
        print(f"Client ID: {client_id} non trouvé dans les données.")