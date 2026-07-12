"""
Page Historique — Consultation des prédictions sauvegardées.
"""

import streamlit as st
import pandas as pd
import os

from utils.auth import (
    check_rgpd_consent,
    check_authentication,
    log_action,
    get_current_user,
)

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("📋 Historique des prédictions")
st.markdown("Consultez et exportez l'historique des scorings effectués.")
st.markdown("---")

log_action(get_current_user(), "Accès page Historique")

HISTORY_PATH = "data/predictions_history.csv"

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
if not os.path.exists(HISTORY_PATH):
    st.info(
        "Aucune prédiction enregistrée pour le moment. "
        "Effectuez un scoring depuis la page **Prédiction** ou **Batch** "
        "pour voir apparaître l'historique ici."
    )
    st.stop()

df = pd.read_csv(HISTORY_PATH)

if df.empty:
    st.info("L'historique est vide.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
st.markdown("### Résumé")

k1, k2, k3 = st.columns(3)
k1.metric("Total d'entrées", f"{len(df):,}")

nb_indiv = (df["type"] == "individuelle").sum() if "type" in df.columns else 0
nb_batch = (df["type"] == "batch").sum() if "type" in df.columns else 0
k2.metric("Prédictions individuelles", f"{nb_indiv:,}")
k3.metric("Scorings batch", f"{nb_batch:,}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
col_f1, col_f2 = st.columns(2)

with col_f1:
    if "type" in df.columns:
        type_filter = st.selectbox("Type de prédiction", ["Tous", "individuelle", "batch"])
        if type_filter != "Tous":
            df = df[df["type"] == type_filter]

with col_f2:
    if "risque" in df.columns:
        risque_filter = st.selectbox("Niveau de risque", ["Tous", "Élevé", "Moyen", "Faible"])
        if risque_filter != "Tous":
            df = df[df["risque"] == risque_filter]

# ---------------------------------------------------------------------------
# Tableau
# ---------------------------------------------------------------------------
st.markdown("### Détail")
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.markdown("---")

csv_data = df.to_csv(index=False, sep=";").encode("utf-8")
st.download_button(
    label="📥 Exporter l'historique (CSV)",
    data=csv_data,
    file_name="historique_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Purge (optionnel)
# ---------------------------------------------------------------------------
with st.expander("🗑️ Purger l'historique"):
    st.warning("Cette action supprimera définitivement toutes les entrées.")
    if st.button("Confirmer la purge", type="secondary"):
        try:
            os.remove(HISTORY_PATH)
            log_action(get_current_user(), "Purge historique des prédictions")
            st.success("Historique purgé.")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("---")
st.caption(
    "⚠️ Sur Streamlit Community Cloud, l'historique est réinitialisé à chaque "
    "redéploiement. Pour une persistance durable, une base de données "
    "(SQLite, PostgreSQL) est recommandée."
)
