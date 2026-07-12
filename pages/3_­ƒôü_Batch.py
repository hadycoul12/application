"""
Page Prédiction batch — Upload CSV, scoring en masse, export.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

from utils.auth import (
    check_rgpd_consent,
    check_authentication,
    log_action,
    get_current_user,
)
from utils.predict import predict_batch, FEATURES_ORDER, SEUIL

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("📁 Prédiction batch")
st.markdown(
    "Uploadez un fichier CSV contenant des réservations à scorer. "
    "Le modèle attribuera un score de risque à chaque dossier."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
with st.expander("📋 Format attendu du fichier CSV"):
    st.markdown("Le fichier doit contenir les colonnes suivantes :")
    cols_display = pd.DataFrame(
        {"Colonne": FEATURES_ORDER, "N°": range(1, len(FEATURES_ORDER) + 1)}
    )
    st.dataframe(cols_display[["N°", "Colonne"]], hide_index=True, use_container_width=True)
    st.info(
        "Les colonnes supplémentaires (ex: `dossier_cle`) seront conservées "
        "dans le fichier de sortie mais ne seront pas utilisées pour le scoring."
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Choisir un fichier CSV",
    type=["csv"],
    help="Encodage UTF-8, séparateur virgule ou point-virgule.",
)

if uploaded is not None:
    log_action(get_current_user(), f"Upload batch : {uploaded.name}")

    # Lecture avec détection du séparateur
    try:
        content = uploaded.getvalue().decode("utf-8")
        sep = ";" if content.count(";") > content.count(",") else ","
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=sep)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

    st.success(f"✅ {len(df):,} dossiers chargés ({len(df.columns)} colonnes)")

    # Vérification des colonnes
    missing = [c for c in FEATURES_ORDER if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {', '.join(missing)}")
        st.stop()

    # --- Aperçu avant scoring ---
    st.markdown("#### Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # --- Scoring ---
    st.markdown("---")

    if st.button("🎯 Scorer tous les dossiers", type="primary", use_container_width=True):
        with st.spinner(f"Scoring de {len(df):,} dossiers en cours..."):
            try:
                df_scored = predict_batch(df)
            except Exception as e:
                st.error(f"Erreur lors du scoring : {e}")
                st.stop()

        log_action(
            get_current_user(),
            f"Batch scoré : {len(df_scored)} dossiers, "
            f"{(df_scored['risque'] == 'Élevé').sum()} à risque élevé",
        )

        st.success(
            f"✅ Scoring terminé — "
            f"**{(df_scored['risque'] == 'Élevé').sum()}** dossiers à risque élevé"
        )

        # --- KPIs du batch ---
        st.markdown("#### Résumé du scoring")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total scoré", f"{len(df_scored):,}")
        b2.metric(
            "🔴 Risque élevé",
            f"{(df_scored['risque'] == 'Élevé').sum():,}",
        )
        b3.metric(
            "🟠 Risque moyen",
            f"{(df_scored['risque'] == 'Moyen').sum():,}",
        )
        b4.metric(
            "🟢 Risque faible",
            f"{(df_scored['risque'] == 'Faible').sum():,}",
        )

        # --- Filtre par niveau de risque ---
        st.markdown("---")
        st.markdown("#### Résultats détaillés")

        risk_filter = st.selectbox(
            "Filtrer par niveau de risque",
            ["Tous", "Élevé", "Moyen", "Faible"],
        )

        df_display = df_scored.copy()
        if risk_filter != "Tous":
            df_display = df_display[df_display["risque"] == risk_filter]

        # Coloriser le risque
        def color_risk(val):
            colors = {"Élevé": "#ffcccc", "Moyen": "#fff3cd", "Faible": "#d4edda"}
            return f"background-color: {colors.get(val, '')}"

        st.dataframe(
            df_display.head(100).style.applymap(
                color_risk, subset=["risque"]
            ),
            use_container_width=True,
            hide_index=True,
        )

        if len(df_display) > 100:
            st.caption(f"Affichage limité aux 100 premiers — {len(df_display)} au total.")

        # --- Export CSV ---
        st.markdown("---")

        csv_export = df_scored.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv_export,
            file_name=f"scoring_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # --- Sauvegarde historique ---
        try:
            history_path = "data/predictions_history.csv"
            summary = {
                "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "utilisateur": get_current_user(),
                "type": "batch",
                "proba_annulation": df_scored["proba_annulation"].mean(),
                "risque": f"{(df_scored['risque'] == 'Élevé').sum()} élevés / {len(df_scored)} total",
            }
            df_hist = pd.DataFrame([summary])

            if os.path.exists(history_path):
                df_hist.to_csv(history_path, mode="a", header=False, index=False)
            else:
                df_hist.to_csv(history_path, mode="w", header=True, index=False)
        except Exception:
            pass
