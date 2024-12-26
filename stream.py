import streamlit as st
import requests
import urllib.parse

st.title("Interface pour l'API de prédiction")

# Paramètres de l'API
API_BASE_URL = "https://projet7v2.onrender.com/predict"

# Champ de saisie pour SK_ID_CURR
sk_id_curr = st.number_input("Entrez le SK_ID_CURR :", value=415174, step=1)  # Valeur par défaut et incrément

if st.button("Obtenir la prédiction"):
    try:
        # Construction de l'URL avec le paramètre
        url_params = {"SK_ID_CURR": sk_id_curr}
        encoded_params = urllib.parse.urlencode(url_params)
        full_url = f"{API_BASE_URL}?{encoded_params}"

        # Envoi de la requête GET
        response = requests.get(full_url)
        response.raise_for_status()  # Gérer les erreurs HTTP
        result = response.json()

        # Affichage du résultat
        st.write("Résultat de l'API :")
        st.json(result) # Affiche le JSON formaté pour une meilleure lisibilité.
        # ou pour un affichage plus personnalisé :
        #if "prediction" in result:
        #    st.write("Prédiction :", result["prediction"])
        #if "probability" in result:
        #    st.write("Probabilité :", result["probability"])

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête à l'API : {e}")
    except ValueError as e:
        st.error(f"Erreur lors du décodage de la réponse JSON : {e}. Vérifiez que l'API renvoie du JSON valide.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite : {e}")