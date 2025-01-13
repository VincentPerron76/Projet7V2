import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Charger le pipeline de production
pipeline = joblib.load("artifacts/production_pipeline.joblib")
# Charger les données des clients à tester
client_data = pd.read_csv("df_data_test_unit_non_scalées.csv", index_col="SK_ID_CURR")

# Calculer les probabilités pour tous les clients
probabilities = pipeline.predict_proba(client_data)[:, 1]

# Trouver les indices des clients avec des prédictions proches de 0 et de 1
clients_classe_0 = client_data.index[probabilities < 0.14]
clients_classe_1 = client_data.index[probabilities >= 0.14]

# Sélectionner 5 clients dans chaque catégorie
clients_classe_0_to_test = clients_classe_0[:5]
clients_classe_1_to_test = clients_classe_1[:5]

# Combiner les deux groupes pour obtenir les 10 clients à tester
client_ids_to_test = list(clients_classe_0_to_test) + list(clients_classe_1_to_test)

# Afficher les clients sélectionnés
print("Clients sélectionnés pour les tests (5 à prédiction 0 et 5 à prédiction 1):")
print(client_ids_to_test)

# Tester ces clients et afficher les prédictions
for client_id in client_ids_to_test:
    if client_id in client_data.index:
        input_data = client_data.loc[[client_id]]
        probabilities = pipeline.predict_proba(input_data)[:, 1]
        seuil_personnalise = 0.14
        prediction = (probabilities >= seuil_personnalise).astype(int)[0]
        print(f"Client ID: {client_id}, Prédiction: {prediction}")
    else:
        print(f"Client ID: {client_id} non trouvé dans les données.")


