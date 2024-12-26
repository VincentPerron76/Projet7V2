import streamlit as st
import requests

# URL de l'API Azure (remplacez par votre URL réelle)
API_URL = "https://<nom-de-votre-service>.azurewebsites.net/predict"

# Titre de l'application
st.title("Interface de Test pour le Modèle Azure")

# Description
st.write("Entrez les données nécessaires pour effectuer une prédiction via l'API hébergée sur Azure.")

# Champs pour les entrées utilisateur
data = {
    "CNT_CHILDREN": st.number_input("CNT_CHILDREN", value=0, step=1, format="%d"),
    "DAYS_BIRTH": st.number_input("DAYS_BIRTH", value=0, step=1, format="%d", help="Nombre de jours depuis la naissance."),
    "DAYS_ID_PUBLISH": st.number_input("DAYS_ID_PUBLISH", value=0, step=1, format="%d", help="Nombre de jours depuis la publication du document."),
    "FLAG_WORK_PHONE": st.number_input("FLAG_WORK_PHONE", value=0, min_value=0, max_value=1, step=1, format="%d", help="1 si téléphone lié au travail, sinon 0."),
    "CNT_FAM_MEMBERS": st.number_input("CNT_FAM_MEMBERS", value=0, step=1, format="%d"),
    "REGION_RATING_CLIENT_W_CITY": st.number_input("REGION_RATING_CLIENT_W_CITY", value=0.0, step=0.001, format="%.3f"),
    "EXT_SOURCE_1": st.number_input("EXT_SOURCE_1", value=0.0, step=0.001, format="%.3f"),
    "EXT_SOURCE_2": st.number_input("EXT_SOURCE_2", value=0.0, step=0.001, format="%.3f"),
    "EXT_SOURCE_3": st.number_input("EXT_SOURCE_3", value=0.0, step=0.001, format="%.3f"),
    "YEARS_BEGINEXPLUATATION_MEDI": st.number_input("YEARS_BEGINEXPLUATATION_MEDI", value=0.0, step=0.001, format="%.3f"),
    "OBS_30_CNT_SOCIAL_CIRCLE": st.number_input("OBS_30_CNT_SOCIAL_CIRCLE", value=0.0, step=0.001, format="%.3f"),
    "DEF_30_CNT_SOCIAL_CIRCLE": st.number_input("DEF_30_CNT_SOCIAL_CIRCLE", value=0.0, step=0.001, format="%.3f"),
    "FLAG_DOCUMENT_3": st.number_input("FLAG_DOCUMENT_3", value=0, min_value=0, max_value=1, step=1, format="%d"),
    "PAYMENT_RATE": st.number_input("PAYMENT_RATE", value=0.0, step=0.001, format="%.3f"),
    "BURO_DAYS_CREDIT_MEAN": st.number_input("BURO_DAYS_CREDIT_MEAN", value=0.0, step=0.001, format="%.3f"),
    "ACTIVE_DAYS_CREDIT_MAX": st.number_input("ACTIVE_DAYS_CREDIT_MAX", value=0.0, step=0.001, format="%.3f"),
    "PREV_APP_CREDIT_PERC_VAR": st.number_input("PREV_APP_CREDIT_PERC_VAR", value=0.0, step=0.001, format="%.3f"),
    "PREV_RATE_DOWN_PAYMENT_MIN": st.number_input("PREV_RATE_DOWN_PAYMENT_MIN", value=0.0, step=0.001, format="%.3f"),
    "PREV_NAME_YIELD_GROUP_high_MEAN": st.number_input("PREV_NAME_YIELD_GROUP_high_MEAN", value=0.0, step=0.001, format="%.3f"),
    "APPROVED_HOUR_APPR_PROCESS_START_MAX": st.number_input("APPROVED_HOUR_APPR_PROCESS_START_MAX", value=0, step=1, format="%d"),
    "POS_MONTHS_BALANCE_MAX": st.number_input("POS_MONTHS_BALANCE_MAX", value=0.0, step=0.001, format="%.3f"),
    "POS_MONTHS_BALANCE_SIZE": st.number_input("POS_MONTHS_BALANCE_SIZE", value=0.0, step=0.001, format="%.3f"),
    "INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE": st.number_input("INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE", value=0.0, step=0.001, format="%.3f"),
    "INSTAL_DPD_MAX": st.number_input("INSTAL_DPD_MAX", value=0.0, step=0.001, format="%.3f"),
    "INSTAL_AMT_PAYMENT_SUM": st.number_input("INSTAL_AMT_PAYMENT_SUM", value=0.0, step=0.001, format="%.3f")
}

# Bouton pour envoyer les données
if st.button("Envoyer les données"):
    try:
        # Envoyer une requête POST à l'API
        response = requests.post(API_URL, json=[data])
        
        # Afficher les résultats
        if response.status_code == 200:
            st.success("Réponse de l'API :")
            api_response = response.json()
            
            # Vérifier et afficher les prédictions et probabilités
            predictions = api_response.get("predictions", [])
            probabilities = api_response.get("probabilities", [])

            st.write(f"Prédictions : {predictions}")
            st.write(f"Probabilités : {probabilities}")
        else:
            st.error(f"Erreur API (Code {response.status_code}) : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")
