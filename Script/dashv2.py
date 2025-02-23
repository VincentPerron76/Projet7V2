import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap  # Import manquant pour les graphiques SHAP

# Définition de l'URL de l'API
#API_URL = "http://localhost:5001/predict"
#API_URL2 = "http://localhost:5001/dataset"

API_URL = "https://projet7v2.onrender.com/predict"
API_URL2 = "https://projet7v2.onrender.com/dataset"

# Définition des couleurs accessibles (respect WCAG)
COLOR_POSITIVE = "#F98B24"  # orange accessible
COLOR_NEGATIVE = "#0000FF"  # bleau accessible
COLOR_BACKGROUND = "##FAFAFA"

# https://app.contrast-finder.org/

# Appliquer un style plus lisible
st.set_page_config(page_title="Prédiction de Crédit", layout="wide")

st.markdown(
    f"""
    <style>
        body {{
            background-color: {COLOR_BACKGROUND};
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .stButton>button {{
            font-size: 18px;
        }}
        .stTextInput>div>div>input {{
            font-size: 16px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Afficher un titre
st.title("Prédiction de Crédit")

# ----------- PARTIE 1 : Réinitialisation -----------

# Ajout d'un titre clair avec une taille de police adaptée (WCAG)
#st.markdown("<h2 style='font-size: 24px; font-weight: bold;'>🔄 Réinitialisation du Dashboard</h2>", unsafe_allow_html=True)

# Bouton pour réinitialiser le dashboard avec un style accessible
if st.button("🔄 Réinitialiser le Dashboard", help="Cliquez pour réinitialiser le tableau de bord"):
    # Réinitialisation des variables dans session_state
    st.session_state.dataset = None  # Efface le dataset
    st.session_state.scatter_fig = None  # Efface le graphique du scatter plot
    st.session_state.feature_x = None  # Réinitialise la sélection de la feature X
    st.session_state.feature_y = None  # Réinitialise la sélection de la feature Y
    st.session_state.SK_ID_CURR = None  # Réinitialise l'ID client
    st.session_state.SK_ID_CURR_REF = None  # Réinitialise l'ID client de référence
    st.session_state.top_features = None  # Réinitialise les contributeurs SHAP affichés
    st.session_state["SK_ID_CURR_input"] = ""  # Réinitialiser explicitement le champ de saisie

    # Relancer l'application pour obtenir une page vierge
    st.rerun()

    # Message de confirmation de la réinitialisation en WCAG-compatible
    st.markdown(
        "<p style='color: green; font-size: 20px; font-weight: bold;'>✅ Le dashboard a été réinitialisé avec succès.</p>", 
        unsafe_allow_html=True
    )

# Vérifier si le dataset est chargé en session_state
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Fonction pour récupérer le dataset depuis l'API
def load_data():
    response = requests.get(API_URL2)
    if response.status_code == 200:
        data = response.json()
        st.session_state.dataset = pd.DataFrame(data)
        st.session_state.dataset.set_index('SK_ID_CURR', inplace=True)
    else:
        st.error("❌ Erreur lors de la récupération des données depuis l'API.")

# Charger le dataset uniquement une fois
if st.session_state.dataset is None:
    load_data()

# Affichage de l'état du dataset pour diagnostic
#print(" Vérification de l'index du dataset :")
#print("st.session_state.dataset.index[:5]")  # Affichage des premiers indices pour débogage

# Initialisation de session_state pour stocker le scatter plot
if "scatter_fig" not in st.session_state:
    st.session_state.scatter_fig = None

#----------------------------------------------------------------------------------------------------


# ----------- PARTIE 2 : Prédiction du client sélectionné -----------
# Injecter du CSS personnalisé pour une taille de police WCAG de 16px
st.markdown("""
    <style>
        .custom-text-input label {
            font-size: 40px !important;
        }
        .custom-text-input input {
            font-size: 40px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Entrée de texte pour l'ID du client avec une taille de police WCAG
SK_ID_CURR = st.text_input(
    "Entrez l'ID du client :", 
    value="", 
    key="SK_ID_CURR_input", 
    help="Veuillez entrer l'ID du client pour effectuer la prédiction de crédit. Par exemple : 12345"
)

st.session_state["SK_ID_CURR"] = SK_ID_CURR 


if SK_ID_CURR:
    try:
        SK_ID_CURR = int(SK_ID_CURR)
        response = requests.get(API_URL, params={"SK_ID_CURR": SK_ID_CURR})
        data = response.json()

        if "error" in data:
            st.error(f"Erreur : {data['error']}")
        else:
            # Affichage du titre et du résultat de la décision dans le même bloc
            #st.subheader(f"Accord de prêt pour le client {SK_ID_CURR} : ")

            # Meilleure accessibilité du résultat
            decision = "👍 ACCEPTÉ" if data["prediction"] == 0 else "❌ REFUSÉ"
            decision_color = COLOR_NEGATIVE if data["prediction"] == 0 else COLOR_POSITIVE
            border_color = COLOR_NEGATIVE if data["prediction"] == 0 else COLOR_POSITIVE

            # Affichage du message avec les couleurs adaptées (compatible WCAG)
            st.markdown(
                f"""
                <div style="border: 4px solid {border_color}; padding: 10px; border-radius: 10px; background-color: #FFFFFF; text-align: center;">
                    <p style="font-size: 18px; font-weight: bold; color: {decision_color}; display: inline;">
                        Accord de prêt pour le client {SK_ID_CURR} : {decision}
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.write("")
            st.write(f"**Pour avoir un accord de prêt le client doit avoir un seuil inférieur à** : {data['seuil_fair_value']:.3}")
            st.write(f"**Le seuil du client demandé est de:** : {data['tot_shape_value']:.3}")

            # Graphique SHAP amélioré avec couleurs accessibles
            top_positive = pd.DataFrame(data["top_positive"])
            top_negative = pd.DataFrame(data["top_negative"]).iloc[::-1]
            top_features = pd.concat([top_positive, top_negative]).set_index("feature")
            st.session_state.top_features = top_features

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [COLOR_POSITIVE if x > 0 else COLOR_NEGATIVE for x in top_features["shap_value"]]
            top_features["shap_value"].plot(kind="barh", ax=ax, color=colors)

            ax.set_xlabel("Contribution des caractéristiques à la décision finale", fontsize=10, fontweight="bold")
            ax.set_ylabel("Caractéritiques du client analysé", fontsize=10, fontweight="bold")
            ax.set_title("Impact des caractéristiques sur la décision", fontsize=14, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)

            st.pyplot(fig,use_container_width=False)

    except ValueError:
        st.error(f"L'ID client doit être un nombre entier.")
#--------------------------------------------------------------------------------






# ----------- PARTIE 3 : Scatter Plot interactif -----------

if "scatter_fig" not in st.session_state:
    st.session_state.scatter_fig = None


if st.session_state.dataset is not None and not st.session_state.dataset.empty:
    st.subheader("Positionnement du client")

    feature_x = st.selectbox(
    "Sélectionnez la première caractéristique",
    st.session_state.dataset.columns,
    key="feature_x",
    help="Veuillez choisir la caractéristique qui sera sur l'axe horizontal"
    )

    feature_y = st.selectbox(
    "Sélectionnez la deuxième caractéristique",
    st.session_state.dataset.columns,
    key="feature_y",
    help="Veuillez choisir la caractéristique qui sera sur l'axe vertical"
    )

    if st.button("Afficher le graphique"):
        if SK_ID_CURR in st.session_state.dataset.index:
            client_data = st.session_state.dataset.loc[SK_ID_CURR]
            fig, ax = plt.subplots(figsize=(6, 4))

            ax.scatter(st.session_state.dataset[feature_x], st.session_state.dataset[feature_y], 
                       alpha=0.5, label="Population générale", color="#000000")
            ax.scatter(client_data[feature_x], client_data[feature_y], 
                       color="#02E346", s=40, label=f"Client {SK_ID_CURR}")

            ax.set_xlabel(feature_x, fontsize=5, fontweight="bold")
            ax.set_ylabel(feature_y, fontsize=5, fontweight="bold")
            ax.set_title(f"Position du client par rapport aux deux caractéristiques sélectionnées", fontsize=10, fontweight="bold")
            ax.legend(fontsize=5)

            st.session_state.scatter_fig = fig

        

        else:
            st.error(f"Le client {SK_ID_CURR} n'existe pas dans la base.")


if st.session_state.scatter_fig is not None:
    st.markdown(
        "<div style='display: flex; justify-content: center;'>",
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.scatter_fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)




# ----------- PARTIE 4 : Comparaison avec client de référence -----------
# Entrée utilisateur pour l'ID du client de référence
SK_ID_CURR_REF = st.text_input(
    "Entrez l'ID du client de référence :", value="", key="client_ref_input",
    help="Veuillez saisir l'ID du client Référence associé au client choisi pour l'étude"
            )

# Stocker la valeur dans session_state pour éviter qu'elle soit perdue
st.session_state["SK_ID_CURR_REF"] = SK_ID_CURR_REF  

if SK_ID_CURR_REF:
    try:
        response_ref = requests.get(API_URL, params={"SK_ID_CURR": SK_ID_CURR_REF})
        data_ref = response_ref.json()

        if "error" in data_ref:
            st.error(f"Erreur : {data_ref['error']}")
        else:
            st.subheader("Comparaison avec Client Référence")

        
            if "top_features" in st.session_state:
                top_features = st.session_state.top_features
                df_shap_value = pd.DataFrame(data_ref.get("detail_des_shap_value", []))

                if not df_shap_value.empty:
                    df_comparison = top_features[['shap_value']].merge(
                        df_shap_value[['feature', 'shap_value']], 
                        on="feature", 
                        suffixes=("_client", "_reference")
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))
                    df_comparison.set_index("feature").plot(kind="barh", ax=ax, color=[COLOR_POSITIVE, COLOR_NEGATIVE])

                    # Récupération des IDs pour la légende
                    id_client = st.session_state.get("SK_ID_CURR", "Client")
                    id_reference = st.session_state.get("SK_ID_CURR_REF", "Référence")

                    # Ajout de la légende avec les ID
                    ax.legend([f"Client {id_client}", f"Client Référence {id_reference}"], fontsize=8)

                    ax.set_title("Comparaison de l'impact des caractéristiques sur la décision de crédit", fontsize=12, fontweight="bold")
                    ax.set_xlabel("Contribution des caractéristiques au seuil final de décision", fontsize=10)
                    ax.set_ylabel("Caractéristiques du client analysé", fontsize=10, fontweight="bold")
                    ax.grid(True, linestyle="--", alpha=0.6)

                    st.pyplot(fig, use_container_width=False)

    except Exception as e:
        st.error(f"Erreur : {str(e)}")