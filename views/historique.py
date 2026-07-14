"""Historique des prédictions enregistrées."""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.auth import current_user, log
from utils.ui import page_header, section

HISTORY = Path(__file__).parent.parent / "data" / "predictions_history.csv"

page_header(
    "Traçabilité",
    "Historique des prédictions",
    "Consultez et exportez l'ensemble des scorings réalisés sur cette instance.",
)

log(current_user(), "Consultation de l'historique")

if not HISTORY.exists():
    st.info(
        "Aucune prédiction enregistrée. Réalisez un scoring depuis la page "
        "**Prédiction** ou **Scoring batch** pour alimenter l'historique.",
        icon=":material/inbox:",
    )
    st.stop()

try:
    df = pd.read_csv(HISTORY, encoding="utf-8")
except Exception as e:
    st.error(f"Impossible de lire l'historique : {e}")
    st.stop()

if df.empty:
    st.info("L'historique est vide.", icon=":material/inbox:")
    st.stop()

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
n_indiv = int((df["type"] == "individuelle").sum()) if "type" in df else 0
n_batch = int((df["type"] == "batch").sum()) if "type" in df else 0
n_high = int((df["risque"] == "Élevé").sum()) if "risque" in df else 0

k1, k2, k3 = st.columns(3)
k1.metric("Entrées", f"{len(df):,}".replace(",", " "))
k2.metric("Prédictions unitaires", f"{n_indiv:,}".replace(",", " "))
k3.metric("Scorings par lot", f"{n_batch:,}".replace(",", " "))

# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
section("Journal")

f1, f2, f3 = st.columns([1, 1, 2])

view = df.copy()

with f1:
    if "type" in df:
        t = st.selectbox("Type", ["Tous"] + sorted(df["type"].dropna().unique().tolist()))
        if t != "Tous":
            view = view[view["type"] == t]

with f2:
    if "risque" in df:
        levels = [r for r in ["Élevé", "Moyen", "Faible"] if r in df["risque"].values]
        r = st.selectbox("Risque", ["Tous"] + levels)
        if r != "Tous":
            view = view[view["risque"] == r]

st.dataframe(
    view.iloc[::-1],
    use_container_width=True,
    hide_index=True,
    height=430,
    column_config={
        "proba_annulation": st.column_config.ProgressColumn(
            "Score", format="%.3f", min_value=0.0, max_value=1.0
        ),
    },
)
st.caption(f"{len(view):,} entrées affichées, les plus récentes en premier.".replace(",", " "))

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
a1, a2 = st.columns(2)

with a1:
    st.download_button(
        "Exporter l'historique (CSV)",
        data=view.to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name="historique_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        icon=":material/download:",
    )

with a2:
    with st.popover("Purger l'historique", use_container_width=True):
        st.markdown("**Cette action est irréversible.**")
        st.caption("Toutes les entrées enregistrées seront supprimées définitivement.")
        if st.button("Confirmer la suppression", type="primary"):
            try:
                HISTORY.unlink()
                log(current_user(), "Purge de l'historique des prédictions")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

st.caption(
    "Sur Streamlit Community Cloud, l'historique est réinitialisé à chaque redéploiement. "
    "Une base de données (SQLite, PostgreSQL) serait nécessaire pour une persistance durable."
)
