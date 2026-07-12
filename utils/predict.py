"""
Chargement du modèle XGBoost et fonctions de prédiction.
Adapter FEATURES_ORDER si les colonnes du modèle changent.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------------
# Configuration des features (à adapter à ton modèle exact)
# ---------------------------------------------------------------------------
# L'ORDRE doit correspondre exactement à celui utilisé pendant l'entraînement.
# Si tu as utilisé un DataFrame pandas pour le fit, l'ordre est celui des colonnes.

FEATURES_ORDER = [
    "canal",
    "est_assure_annulation",
    "anticipation_jours",
    "nb_dossiers_anterieurs",
    "est_solo",
    "assure_x_anticip",
    "a_prestations",
    "nb_produits_total",
    "region_destination",
    "device_resa",
    "theme_station",
    "periode_depart",
    "duree_sejour",
    "groupe_fournisseur",
    "nb_campagnes_recues",
    "recence_email_jours",
    "est_dans_crm",
]

# Labels lisibles pour l'affichage dans l'app
FEATURES_LABELS = {
    "canal": "Canal de réservation",
    "est_assure_annulation": "Assuré Flex (oui/non)",
    "anticipation_jours": "Anticipation (jours)",
    "nb_dossiers_anterieurs": "Nb dossiers antérieurs",
    "est_solo": "Réservation solo",
    "assure_x_anticip": "Interaction assurance × anticipation",
    "a_prestations": "A des prestations",
    "nb_produits_total": "Nb produits total",
    "region_destination": "Région destination",
    "device_resa": "Device réservation",
    "theme_station": "Thème station",
    "periode_depart": "Période de départ",
    "duree_sejour": "Durée séjour (nuits)",
    "groupe_fournisseur": "Groupe fournisseur",
    "nb_campagnes_recues": "Nb campagnes email reçues",
    "recence_email_jours": "Récence dernier email (jours)",
    "est_dans_crm": "Présent dans le CRM",
}

# Seuil de décision retenu après optimisation
SEUIL = 0.50

# Chemin du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_optimise.joblib")


# ---------------------------------------------------------------------------
# Chargement du modèle (cache Streamlit pour ne charger qu'une fois)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Charge le modèle XGBoost depuis le fichier joblib."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Modèle introuvable : `{MODEL_PATH}`. "
            "Place ton fichier `xgb_optimise.joblib` dans le dossier `models/`."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
def predict_single(features_dict: dict) -> dict:
    """
    Prédiction pour un dossier unique.

    Parameters
    ----------
    features_dict : dict
        Dictionnaire {nom_feature: valeur} pour un seul dossier.

    Returns
    -------
    dict avec clés : proba, risque, label_couleur
    """
    model = load_model()
    df = pd.DataFrame([features_dict])[FEATURES_ORDER]
    proba = model.predict_proba(df)[:, 1][0]

    return _format_prediction(proba)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prédiction pour un DataFrame de dossiers.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir toutes les colonnes de FEATURES_ORDER.

    Returns
    -------
    DataFrame enrichi avec proba_annulation, risque, couleur
    """
    model = load_model()

    # Vérifier les colonnes manquantes
    missing = [c for c in FEATURES_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    X = df[FEATURES_ORDER]
    probas = model.predict_proba(X)[:, 1]

    df_result = df.copy()
    df_result["proba_annulation"] = probas
    df_result["risque"] = df_result["proba_annulation"].apply(_get_risk_level)
    df_result = df_result.sort_values("proba_annulation", ascending=False)

    return df_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_prediction(proba: float) -> dict:
    """Formate une probabilité en résultat structuré."""
    risk = _get_risk_level(proba)
    color_map = {"Élevé": "🔴", "Moyen": "🟠", "Faible": "🟢"}

    return {
        "proba": round(proba, 4),
        "proba_pct": f"{proba * 100:.1f}%",
        "risque": risk,
        "emoji": color_map.get(risk, "⚪"),
    }


def _get_risk_level(proba: float) -> str:
    """Classe un score en niveau de risque."""
    if proba >= SEUIL:
        return "Élevé"
    elif proba >= 0.30:
        return "Moyen"
    else:
        return "Faible"
