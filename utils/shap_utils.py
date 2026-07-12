"""
Utilitaires SHAP pour l'interprétabilité des prédictions.
"""

import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
from utils.predict import load_model, FEATURES_ORDER, FEATURES_LABELS


# ---------------------------------------------------------------------------
# Explainer (cache pour ne pas recalculer à chaque prédiction)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_explainer():
    """Crée un TreeExplainer SHAP pour le modèle XGBoost."""
    model = load_model()
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Waterfall plot pour une prédiction individuelle
# ---------------------------------------------------------------------------
def plot_shap_waterfall(features_dict: dict):
    """
    Génère un waterfall plot SHAP pour un dossier unique.

    Parameters
    ----------
    features_dict : dict
        {nom_feature: valeur} pour un seul dossier.

    Returns
    -------
    matplotlib.figure.Figure
    """
    explainer = get_explainer()
    df = pd.DataFrame([features_dict])[FEATURES_ORDER]

    # Calcul des SHAP values
    shap_values = explainer.shap_values(df)

    # Renommer les features pour l'affichage
    feature_names_display = [FEATURES_LABELS.get(f, f) for f in FEATURES_ORDER]

    # Créer l'objet Explanation
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=df.values[0],
        feature_names=feature_names_display,
    )

    # Trier par valeur absolue pour le waterfall
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title("Décomposition SHAP — contribution de chaque variable", fontsize=12)
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Force plot (version compacte inline)
# ---------------------------------------------------------------------------
def get_shap_explanation_text(features_dict: dict) -> str:
    """
    Génère une explication textuelle des principaux facteurs de risque.

    Parameters
    ----------
    features_dict : dict

    Returns
    -------
    str : texte explicatif en français
    """
    explainer = get_explainer()
    df = pd.DataFrame([features_dict])[FEATURES_ORDER]
    shap_values = explainer.shap_values(df)

    # Top 5 features par impact absolu
    impacts = list(zip(FEATURES_ORDER, shap_values[0]))
    impacts_sorted = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:5]

    lines = []
    for feat, val in impacts_sorted:
        label = FEATURES_LABELS.get(feat, feat)
        direction = "augmente" if val > 0 else "diminue"
        lines.append(f"• **{label}** {direction} le risque ({val:+.3f})")

    return "\n".join(lines)
