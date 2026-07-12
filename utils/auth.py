"""
Gestion de l'authentification et du journal d'audit RGPD.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.csv")


# ---------------------------------------------------------------------------
# Bannière de consentement RGPD (bloquante)
# ---------------------------------------------------------------------------
def check_rgpd_consent():
    """Affiche la bannière RGPD si le consentement n'a pas été donné."""
    if st.session_state.get("rgpd_consent"):
        return True

    st.markdown("---")
    st.markdown(
        """
        ### 🔒 Protection des données personnelles

        Cette application utilise un **modèle de scoring prédictif** entraîné
        sur des données de réservation **anonymisées**. Aucune donnée personnelle
        identifiante (nom, email, téléphone, adresse) n'est collectée,
        stockée ou traitée par cette application.

        **Finalité du traitement :** prévention des annulations de réservation
        par identification des dossiers à risque, dans le cadre de l'intérêt
        légitime de l'entreprise (article 6.1.f du RGPD).

        **Données traitées :** variables structurelles de réservation uniquement
        (canal, condition d'annulation, anticipation, durée de séjour, etc.).

        **Durée de conservation :** session uniquement — les prédictions ne sont
        pas conservées au-delà de la session en cours.

        En cliquant sur « J'accepte », vous reconnaissez avoir pris connaissance
        de ces informations.
        """
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("✅ J'accepte", use_container_width=True, type="primary"):
            st.session_state["rgpd_consent"] = True
            st.session_state["rgpd_consent_at"] = datetime.now().isoformat()
            log_action("anonyme", "Consentement RGPD accepté")
            st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
def check_authentication():
    """Vérifie si l'utilisateur est authentifié, sinon affiche le login."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("## 🔐 Connexion")
    st.markdown("Accès réservé aux gestionnaires de réservation Maeva.")

    with st.form("login_form"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary")

    if submitted:
        valid_user = st.secrets["credentials"]["username"]
        valid_pass = st.secrets["credentials"]["password"]

        if username == valid_user and password == valid_pass:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["login_at"] = datetime.now().isoformat()
            log_action(username, "Connexion réussie")
            st.rerun()
        else:
            log_action(username or "inconnu", "Tentative de connexion échouée")
            st.error("Identifiant ou mot de passe incorrect.")

    st.stop()


# ---------------------------------------------------------------------------
# Journal d'audit
# ---------------------------------------------------------------------------
def log_action(user: str, action: str):
    """Enregistre une action horodatée dans le journal d'audit."""
    entry = {
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "utilisateur": user,
        "action": action,
    }
    df_entry = pd.DataFrame([entry])

    try:
        if os.path.exists(AUDIT_LOG_PATH):
            df_entry.to_csv(AUDIT_LOG_PATH, mode="a", header=False, index=False)
        else:
            os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
            df_entry.to_csv(AUDIT_LOG_PATH, mode="w", header=True, index=False)
    except Exception:
        pass  # En cas d'erreur d'écriture (permissions), on ne bloque pas l'app


def get_audit_log() -> pd.DataFrame:
    """Charge le journal d'audit."""
    if os.path.exists(AUDIT_LOG_PATH):
        return pd.read_csv(AUDIT_LOG_PATH)
    return pd.DataFrame(columns=["horodatage", "utilisateur", "action"])


def get_current_user() -> str:
    """Retourne le nom de l'utilisateur connecté."""
    return st.session_state.get("username", "inconnu")
