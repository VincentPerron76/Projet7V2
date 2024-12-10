import streamlit as st
import requests

# URL de l'API
API_URL = "http://127.0.0.1:5001/predict"

# Titre de l'application
st.title("Interface de Test de l'API")

# Description
st.write("Entrez les données nécessaires pour effectuer une prédiction via l'API.")

# Créez des champs pour les entrées utilisateur
data = {
"CNT_CHILDREN": st.number_input("CNT_CHILDREN", value=None, step=1, format="%d"),
"DAYS_BIRTH": st.number_input("DAYS_BIRTH", value=None, step=1, format="%d", help="Nombre d'années sous forme d'entier, séparateur de milliers."),
"DAYS_ID_PUBLISH": st.number_input("DAYS_ID_PUBLISH", value=None, step=1, format="%d", help="Nombre d'années sous forme d'entier, séparateur de milliers."),
"FLAG_WORK_PHONE": st.number_input("FLAG_WORK_PHONE", value=None, min_value=0, max_value=1, step=1, format="%d", help="1 si le numéro de téléphone est associé au travail, sinon 0."),
"CNT_FAM_MEMBERS": st.number_input("CNT_FAM_MEMBERS", value=None, step=1, format="%d"),
"REGION_RATING_CLIENT_W_CITY": st.number_input("REGION_RATING_CLIENT_W_CITY", value=None, step=0.001, format="%.3f"),
"EXT_SOURCE_1": st.number_input("EXT_SOURCE_1", value=None, step=0.001, format="%.3f"),
"EXT_SOURCE_2": st.number_input("EXT_SOURCE_2", value=None, step=0.001, format="%.3f"),
"EXT_SOURCE_3": st.number_input("EXT_SOURCE_3", value=None, step=0.001, format="%.3f"),
"YEARS_BEGINEXPLUATATION_MEDI": st.number_input("YEARS_BEGINEXPLUATATION_MEDI", value=None, step=0.001, format="%.3f"),
"OBS_30_CNT_SOCIAL_CIRCLE": st.number_input("OBS_30_CNT_SOCIAL_CIRCLE", value=None, step=0.001, format="%.3f"),
"DEF_30_CNT_SOCIAL_CIRCLE": st.number_input("DEF_30_CNT_SOCIAL_CIRCLE", value=None, step=0.001, format="%.3f"),
"FLAG_DOCUMENT_3": st.number_input("FLAG_DOCUMENT_3", value=None, min_value=0, max_value=1, step=1, format="%d", help="1 si le document existe, sinon 0."),
"PAYMENT_RATE": st.number_input("PAYMENT_RATE", value=None, step=0.001, format="%.3f"),
"BURO_DAYS_CREDIT_MEAN": st.number_input("BURO_DAYS_CREDIT_MEAN", value=None, step=0.001, format="%.3f"),
"ACTIVE_DAYS_CREDIT_MAX": st.number_input("ACTIVE_DAYS_CREDIT_MAX", value=None, step=0.001, format="%.3f"),
"PREV_APP_CREDIT_PERC_VAR": st.number_input("PREV_APP_CREDIT_PERC_VAR", value=None, step=0.001, format="%.3f"),
"PREV_RATE_DOWN_PAYMENT_MIN": st.number_input("PREV_RATE_DOWN_PAYMENT_MIN", value=None, step=0.001, format="%.3f"),
"PREV_NAME_YIELD_GROUP_high_MEAN": st.number_input("PREV_NAME_YIELD_GROUP_high_MEAN", value=None, step=0.001, format="%.3f"),
"APPROVED_HOUR_APPR_PROCESS_START_MAX": st.number_input("APPROVED_HOUR_APPR_PROCESS_START_MAX", value=None, step=1, format="%d", help="Heure maximale d'approbation de l'application."),
"POS_MONTHS_BALANCE_MAX": st.number_input("POS_MONTHS_BALANCE_MAX", value=None, step=0.001, format="%.3f"),
"POS_MONTHS_BALANCE_SIZE": st.number_input("POS_MONTHS_BALANCE_SIZE", value=None, step=0.001, format="%.3f"),
"INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE": st.number_input("INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE", value=None, step=0.001, format="%.3f"),
"INSTAL_DPD_MAX": st.number_input("INSTAL_DPD_MAX", value=None, step=0.001, format="%.3f"),
"INSTAL_AMT_PAYMENT_SUM": st.number_input("INSTAL_AMT_PAYMENT_SUM", value=None, step=0.001, format="%.3f")
}




# Bouton pour envoyer les données
if st.button("Envoyer les données"):
    try:
        # Envoyer une requête POST à l'API
        response = requests.post(API_URL, json=[data])
        
        # Afficher les résultats
        if response.status_code == 200:
            st.success("Réponse de l'API :")
            # Récupérer la réponse JSON
            api_response = response.json()
            
            # Assurez-vous que la réponse contient bien des listes et non des dictionnaires avec des index bizarres
            predictions = api_response.get("predictions", [])
            probabilities = api_response.get("probabilities", [])
            
            # Vérifiez la structure avant d'afficher
            if isinstance(predictions, list):
                st.write("Prédictions :", predictions)
            else:
                st.write("Erreur dans la structure des prédictions.")
            
            if isinstance(probabilities, list):
                st.write("Probabilités :", probabilities)
            else:
                st.write("Erreur dans la structure des probabilités.")
        else:
            st.error(f"Erreur API (Code {response.status_code})")
            st.text(response.text)
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")