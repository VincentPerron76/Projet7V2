import streamlit as st
import requests
import shap
import pandas as pd
import urllib.parse

# Titre du dashboard
st.title("Dashboard Prédiction de Prêt")

# Paramètres de l'API
API_BASE_URL = "http://127.0.0.1:5001/predict"  # L'URL de votre API Flask

# Champ de saisie pour SK_ID_CURR
sk_id_curr = st.number_input("Entrez le SK_ID_CURR :", value=180213, step=1)  # Valeur par défaut et incrément

# Bouton pour obtenir la prédiction
if st.button("Obtenir la prédiction"):

    try:
        # Construction de l'URL avec le paramètre SK_ID_CURR
        url_params = {"SK_ID_CURR": sk_id_curr}
        encoded_params = urllib.parse.urlencode(url_params)
        full_url = f"{API_BASE_URL}?{encoded_params}"

        # Envoi de la requête GET
        response = requests.get(full_url)
        response.raise_for_status()  # Gérer les erreurs HTTP

        # Vérifier si l'API renvoie un message d'erreur spécifique pour le client inconnu
        result = response.json()

        if "error" in result:
            # Si un message d'erreur est trouvé dans la réponse 
            st.error(f"Erreur : {result['error']}")
        else:
            # Affichage du résultat de l'API
            st.write("Résultat de la prédiction :")
            st.json(result)  # Affiche le JSON formaté pour une meilleure lisibilité.

            # Récupérer les données d'entrée pour calculer les valeurs SHAP
            input_data = pd.DataFrame(result['input_data'])  # Adapter selon la structure de votre réponse
            model = result['model']  # Assurez-vous que le modèle est aussi dans la réponse

            # Initialiser l'explainer SHAP avec le modèle récupéré depuis l'API
            explainer = shap.TreeExplainer(model)  # Adaptez si c'est un autre type de modèle
            shap_values = explainer.shap_values(input_data)

            # Afficher les valeurs SHAP
            st.write("Valeurs SHAP pour l'individu :")
            st.write(shap_values)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête à l'API : {e}")
    except ValueError as e:
        st.error(f"Erreur lors du décodage de la réponse JSON : {e}. Vérifiez que l'API renvoie du JSON valide.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite : {e}")


