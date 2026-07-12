"""
Application Streamlit — Scoring prédictif du risque d'annulation
Maeva / Pierre & Vacances — Mémoire Master 2 Data & IA
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration de la page (doit être le PREMIER appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Scoring Annulation — Maeva",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports après set_page_config
# ---------------------------------------------------------------------------
from utils.auth import check_rgpd_consent, check_authentication, get_current_user

# ---------------------------------------------------------------------------
# Étape 1 : Consentement RGPD (bloquant)
# ---------------------------------------------------------------------------
check_rgpd_consent()

# ---------------------------------------------------------------------------
# Étape 2 : Authentification (bloquant)
# ---------------------------------------------------------------------------
check_authentication()

# ---------------------------------------------------------------------------
# Étape 3 : Page d'accueil (si authentifié)
# ---------------------------------------------------------------------------
user = get_current_user()

st.sidebar.markdown(f"👤 Connecté : **{user}**")
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Déconnexion"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- Contenu principal ---
st.title("🎯 Scoring prédictif — Risque d'annulation")
st.markdown("---")

st.markdown(
    """
    Bienvenue sur l'outil de scoring prédictif des annulations de réservation.

    Cette application permet aux gestionnaires de réservation de :
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### 📊 Comprendre
        Visualisez les tendances d'annulation
        par canal, condition, période et région
        via le **Dashboard**.
        """
    )

with col2:
    st.markdown(
        """
        ### 🎯 Prédire
        Scorez un dossier individuel ou un
        lot de réservations pour identifier
        les dossiers à risque.
        """
    )

with col3:
    st.markdown(
        """
        ### 💰 Agir
        Simulez l'impact business d'une
        campagne de rétention ciblée
        sur les dossiers alertés.
        """
    )

st.markdown("---")

st.info(
    "👈 Utilisez la **barre latérale** pour naviguer entre les pages. "
    "Commencez par le **Dashboard** pour explorer vos données."
)

# --- Métriques clés du modèle ---
st.markdown("### Performances du modèle déployé")

m1, m2, m3, m4 = st.columns(4)
m1.metric("PR-AUC", "0.2626", help="Métrique principale — 3.6× la baseline aléatoire")
m2.metric("Rappel", "62.7%", help="Part des annulations réelles détectées")
m3.metric("Précision", "16.8%", help="Part des alertes qui sont de vraies annulations")
m4.metric("Seuil retenu", "0.50", help="Seuil optimisé (F1, rappel ≥ 60%)")

st.caption(
    "Modèle XGBoost optimisé par RandomizedSearchCV (100 itérations). "
    "Pondération native des classes, sans SMOTE. "
    "Évalué sur 20% de holdout stratifié (~62 000 dossiers)."
)
