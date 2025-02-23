import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Définition de l'URL de l'API
API_URL = "http://localhost:5001/predict"
API_URL2 = "http://localhost:5001/dataset"

# Titre de l'application
st.title("Dashboard Prédiction de Crédit")

# Entrée utilisateur pour l'ID client
SK_ID_CURR = st.text_input("Entrez l'ID du client :", "")
SK_ID_CURR_REF = st.text_input("Entrez l'ID du client de référence :", "")

# Stocker l'état du dataset dans la session pour éviter de le recharger à chaque changement
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Fonction pour récupérer le dataset depuis l'API
def load_data():
    response = requests.get(API_URL2)
    if response.status_code == 200:
        data = response.json()
        st.session_state.dataset = pd.DataFrame(data)
    else:
        st.error("Erreur lors de la récupération des données depuis l'API")

# Si le dataset n'est pas encore chargé, le charger
if st.session_state.dataset is None:
    load_data()

# Vérification si le dataset est vide
if st.session_state.dataset.empty:
    st.warning("Le dataset est vide. Vérifiez l'API.")
else:
    # Sélection des features pour le scatter plot
    st.subheader("Visualisation Scatter Plot")

    feature_x = st.selectbox("Sélectionnez la première feature (axe X)", st.session_state.dataset.columns, key="feature_x")
    feature_y = st.selectbox("Sélectionnez la deuxième feature (axe Y)", st.session_state.dataset.columns, key="feature_y")

    # Bouton pour afficher le scatter plot
    if st.button("Afficher le scatter plot"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(st.session_state.dataset[feature_x], st.session_state.dataset[feature_y], alpha=0.5, label="Population générale")

        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"Comparaison {feature_x} vs {feature_y}")
        ax.legend()
        st.pyplot(fig)

# Récupération des données SHAP pour l'ID client sélectionné
if SK_ID_CURR:
    try:
        response = requests.get(API_URL, params={"SK_ID_CURR": SK_ID_CURR})
        data = response.json()

        if "error" in data:
            st.error(f"Erreur : {data['error']}")
        else:
            # Affichage des résultats SHAP pour le client
            st.subheader(f"Prédiction pour le client {SK_ID_CURR}")
            st.write(f"**Fair Value du modèle** : {data['base_value']:.3}")
            st.write(f"**Seuil d'acception du prêt** : {data['seuil_fair_value']:.3}")
            st.write(f"**Shap Value du client** : {data['tot_shape_value']:.3}")
            st.write(f"**Décision du modèle** : {'✅ Approuvé' if data['prediction'] == 0 else '❌ Refusé'}")

            # Récupération des contributeurs SHAP
            top_positive = pd.DataFrame(data["top_positive"])
            top_negative = pd.DataFrame(data["top_negative"])

            # Création du graphique en barres
            fig, ax = plt.subplots(figsize=(8, 6))
            top_features = pd.concat([top_positive, top_negative]).set_index("feature")
            top_features["shap_value"].plot(kind="barh", ax=ax, color=["red" if x > 0 else "green" for x in top_features["shap_value"]])

            ax.set_xlabel("Valeur SHAP")
            ax.set_ylabel("Feature")
            ax.set_title("Top 7 contributeurs positifs et négatifs")
            ax.grid(True, linestyle="--", alpha=0.6)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {str(e)}")

# Récupérer les données du client de référence pour comparer
if SK_ID_CURR_REF:
    try:
        response_ref = requests.get(API_URL, params={"SK_ID_CURR": SK_ID_CURR_REF})
        data_ref = response_ref.json()

        # Comparaison des résultats avec le client de référence
        st.subheader("Comparaison avec client Référence")

        # Récupération des SHAP values
        df_shap_value = pd.DataFrame(data_ref["detail_des_shap_value"])

        # Fusionner et afficher la comparaison des valeurs SHAP
        df_comparison = top_features[['shap_value']].merge(
            df_shap_value[['feature', 'shap_value']], 
            on="feature", 
            suffixes=("_client", "_reference")
        )

        # Création du graphique de comparaison
        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.4
        index = range(len(df_comparison))
        ax.barh(index, df_comparison["shap_value_client"], bar_width, label=f"Client {SK_ID_CURR}", color="blue")
        ax.barh([i + bar_width for i in index], df_comparison["shap_value_reference"], bar_width, label=f"Client Référence {SK_ID_CURR_REF}", color="orange")

        ax.set_yticks([i + bar_width / 2 for i in index])
        ax.set_yticklabels(df_comparison["feature"])
        ax.set_xlabel("Valeur SHAP")
        ax.set_title("Comparaison des Valeurs SHAP entre Clients")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'appel API pour le client de référence : {str(e)}")