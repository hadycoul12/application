"""
Page Prédiction individuelle — Formulaire + score + SHAP waterfall.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.auth import (
    check_rgpd_consent,
    check_authentication,
    log_action,
    get_current_user,
)
from utils.predict import predict_single, FEATURES_ORDER, FEATURES_LABELS, SEUIL
from utils.shap_utils import plot_shap_waterfall, get_shap_explanation_text

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("🎯 Prédiction individuelle")
st.markdown(
    "Renseignez les caractéristiques d'une réservation pour obtenir "
    "le score de risque d'annulation et son explication."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Formulaire de saisie
# ---------------------------------------------------------------------------
st.markdown("### Caractéristiques de la réservation")

col1, col2, col3 = st.columns(3)

with col1:
    canal = st.selectbox(
        FEATURES_LABELS["canal"],
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Site Maeva",
            1: "Booking",
            2: "Téléphone",
            3: "Agence",
            4: "Partenaire",
            5: "Autre",
        }.get(x, str(x)),
        help="Canal par lequel la réservation a été effectuée.",
    )

    est_assure = st.selectbox(
        FEATURES_LABELS["est_assure_annulation"],
        options=[0, 1],
        format_func=lambda x: "Oui (Flex)" if x == 1 else "Non",
        help="Le client a-t-il une assurance annulation Flex ?",
    )

    anticipation = st.number_input(
        FEATURES_LABELS["anticipation_jours"],
        min_value=0,
        max_value=365,
        value=60,
        step=1,
        help="Nombre de jours entre la réservation et le début du séjour.",
    )

    nb_dossiers = st.number_input(
        FEATURES_LABELS["nb_dossiers_anterieurs"],
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Nombre de réservations antérieures du client.",
    )

with col2:
    est_solo = st.selectbox(
        FEATURES_LABELS["est_solo"],
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )

    a_prestations = st.selectbox(
        FEATURES_LABELS["a_prestations"],
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )

    nb_produits = st.number_input(
        FEATURES_LABELS["nb_produits_total"],
        min_value=1,
        max_value=20,
        value=1,
        step=1,
    )

    duree_sejour = st.number_input(
        FEATURES_LABELS["duree_sejour"],
        min_value=1,
        max_value=30,
        value=7,
        step=1,
    )

with col3:
    region_dest = st.number_input(
        FEATURES_LABELS["region_destination"],
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Code numérique de la région de destination.",
    )

    device_resa = st.selectbox(
        FEATURES_LABELS["device_resa"],
        options=[0, 1, 2],
        format_func=lambda x: {0: "Desktop", 1: "Mobile", 2: "Tablette"}.get(
            x, str(x)
        ),
    )

    theme_station = st.number_input(
        FEATURES_LABELS["theme_station"],
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    periode_depart = st.number_input(
        FEATURES_LABELS["periode_depart"],
        min_value=0,
        max_value=12,
        value=6,
        step=1,
        help="Mois de départ (1=janvier, 12=décembre).",
    )

# --- Features supplémentaires (section dépliable) ---
with st.expander("📧 Variables email (optionnel)"):
    st.caption("Renseignez ces champs si le client est dans le CRM Batch.")

    est_dans_crm = st.selectbox(
        FEATURES_LABELS["est_dans_crm"],
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )

    nb_campagnes = st.number_input(
        FEATURES_LABELS["nb_campagnes_recues"],
        min_value=0,
        max_value=200,
        value=0,
        step=1,
    )

    recence_email = st.number_input(
        FEATURES_LABELS["recence_email_jours"],
        min_value=0,
        max_value=365,
        value=0,
        step=1,
    )

with st.expander("⚙️ Variables dérivées"):
    groupe_fournisseur = st.number_input(
        FEATURES_LABELS["groupe_fournisseur"],
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

# Calcul automatique de l'interaction
assure_x_anticip = est_assure * anticipation

# ---------------------------------------------------------------------------
# Construction du dictionnaire de features
# ---------------------------------------------------------------------------
features = {
    "canal": canal,
    "est_assure_annulation": est_assure,
    "anticipation_jours": anticipation,
    "nb_dossiers_anterieurs": nb_dossiers,
    "est_solo": est_solo,
    "assure_x_anticip": assure_x_anticip,
    "a_prestations": a_prestations,
    "nb_produits_total": nb_produits,
    "region_destination": region_dest,
    "device_resa": device_resa,
    "theme_station": theme_station,
    "periode_depart": periode_depart,
    "duree_sejour": duree_sejour,
    "groupe_fournisseur": groupe_fournisseur,
    "nb_campagnes_recues": nb_campagnes,
    "recence_email_jours": recence_email,
    "est_dans_crm": est_dans_crm,
}

# ---------------------------------------------------------------------------
# Bouton de scoring
# ---------------------------------------------------------------------------
st.markdown("---")

if st.button("🎯 Scorer ce dossier", type="primary", use_container_width=True):
    log_action(get_current_user(), "Prédiction individuelle lancée")

    with st.spinner("Calcul du score en cours..."):
        try:
            result = predict_single(features)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.stop()

    # --- Affichage du résultat ---
    st.markdown("---")
    st.markdown("### Résultat du scoring")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Probabilité d'annulation",
            result["proba_pct"],
        )

    with r2:
        color_map = {"Élevé": "red", "Moyen": "orange", "Faible": "green"}
        risk_color = color_map.get(result["risque"], "gray")
        st.markdown(
            f"**Niveau de risque**<br>"
            f"<span style='font-size:2rem; color:{risk_color};'>"
            f"{result['emoji']} {result['risque']}</span>",
            unsafe_allow_html=True,
        )

    with r3:
        st.metric("Seuil appliqué", f"{SEUIL:.2f}")

    # --- Jauge visuelle ---
    proba_val = result["proba"]
    bar_color = risk_color
    st.markdown(
        f"""
        <div style="background:#eee; border-radius:8px; height:30px; width:100%; position:relative; margin:10px 0;">
            <div style="background:{bar_color}; width:{proba_val*100:.1f}%; height:100%;
                        border-radius:8px; transition: width 0.5s;"></div>
            <div style="position:absolute; left:{SEUIL*100}%; top:0; height:100%;
                        border-left:3px dashed #333; z-index:1;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888;">
            <span>0%</span>
            <span>Seuil ({SEUIL*100:.0f}%)</span>
            <span>100%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Explication SHAP textuelle ---
    st.markdown("---")
    st.markdown("### 🔍 Pourquoi ce score ?")

    try:
        explanation_text = get_shap_explanation_text(features)
        st.markdown(explanation_text)
    except Exception as e:
        st.warning(f"Explication textuelle indisponible : {e}")

    # --- Waterfall SHAP ---
    st.markdown("### 📊 Décomposition SHAP")

    try:
        fig = plot_shap_waterfall(features)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Graphique SHAP indisponible : {e}")

    # --- Sauvegarde dans l'historique ---
    try:
        import os

        history_path = "data/predictions_history.csv"
        entry = {
            "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "utilisateur": get_current_user(),
            "type": "individuelle",
            "proba_annulation": result["proba"],
            "risque": result["risque"],
            **features,
        }
        df_entry = pd.DataFrame([entry])

        if os.path.exists(history_path):
            df_entry.to_csv(history_path, mode="a", header=False, index=False)
        else:
            df_entry.to_csv(history_path, mode="w", header=True, index=False)

        log_action(
            get_current_user(),
            f"Prédiction individuelle sauvegardée (proba={result['proba']:.3f}, risque={result['risque']})",
        )
    except Exception:
        pass  # Ne pas bloquer l'app si la sauvegarde échoue

# Nécessaire pour plt.close() dans le bloc SHAP
import matplotlib.pyplot as plt
