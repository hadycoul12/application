"""
Page Dashboard — Vue d'ensemble du portefeuille de réservations.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.auth import check_rgpd_consent, check_authentication, log_action, get_current_user

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("📊 Dashboard — Vue d'ensemble")
st.markdown("---")

log_action(get_current_user(), "Accès page Dashboard")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/sample_dataset.csv")
        return df
    except FileNotFoundError:
        return None


df = load_data()

if df is None:
    st.warning(
        "⚠️ Fichier `data/sample_dataset.csv` introuvable. "
        "Placez un échantillon anonymisé du dataset dans le dossier `data/` "
        "pour activer le dashboard."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Filtres dans la sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filtres")

# Filtre canal
if "canal" in df.columns:
    canaux = ["Tous"] + sorted(df["canal"].dropna().unique().tolist())
    canal_filter = st.sidebar.selectbox("Canal de réservation", canaux)
    if canal_filter != "Tous":
        df = df[df["canal"] == canal_filter]

# Filtre condition annulation
if "est_assure_annulation" in df.columns:
    assure_filter = st.sidebar.selectbox(
        "Condition Flex",
        ["Tous", "Assuré (Flex)", "Non assuré"],
    )
    if assure_filter == "Assuré (Flex)":
        df = df[df["est_assure_annulation"] == 1]
    elif assure_filter == "Non assuré":
        df = df[df["est_assure_annulation"] == 0]

st.sidebar.markdown(f"**{len(df):,}** dossiers affichés")

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
target_col = "y_annulation"

if target_col not in df.columns:
    st.error(f"Colonne cible `{target_col}` introuvable dans le dataset.")
    st.stop()

total = len(df)
nb_annul = int(df[target_col].sum())
taux_annul = nb_annul / total * 100 if total > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total dossiers", f"{total:,}")
k2.metric("Annulations", f"{nb_annul:,}")
k3.metric("Taux d'annulation", f"{taux_annul:.1f}%")
k4.metric("Dossiers maintenus", f"{total - nb_annul:,}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

# --- Taux d'annulation par canal ---
with col_left:
    st.markdown("#### Taux d'annulation par canal")
    if "canal" in df.columns:
        taux_canal = (
            df.groupby("canal")[target_col]
            .mean()
            .sort_values(ascending=False)
            * 100
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(taux_canal.index, taux_canal.values, color="#E8593C", alpha=0.8)
        ax.set_xlabel("Taux d'annulation (%)")
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Colonne `canal` non disponible.")

# --- Taux d'annulation par condition Flex ---
with col_right:
    st.markdown("#### Assuré Flex vs Non assuré")
    if "est_assure_annulation" in df.columns:
        taux_flex = (
            df.groupby("est_assure_annulation")[target_col]
            .agg(["mean", "count"])
        )
        taux_flex["mean"] = taux_flex["mean"] * 100
        taux_flex.index = taux_flex.index.map({0: "Non assuré", 1: "Assuré (Flex)"})

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            taux_flex.index,
            taux_flex["mean"],
            color=["#3B8BD4", "#E8593C"],
            alpha=0.8,
        )
        ax.set_ylabel("Taux d'annulation (%)")
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Colonne `est_assure_annulation` non disponible.")

st.markdown("---")

# --- Distribution de l'anticipation ---
st.markdown("#### Distribution de l'anticipation (jours avant le séjour)")
if "anticipation_jours" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(
        [
            df[df[target_col] == 0]["anticipation_jours"].dropna(),
            df[df[target_col] == 1]["anticipation_jours"].dropna(),
        ],
        bins=50,
        label=["Maintenu", "Annulé"],
        color=["#3B8BD4", "#E8593C"],
        alpha=0.7,
        stacked=False,
    )
    ax.set_xlabel("Anticipation (jours)")
    ax.set_ylabel("Nombre de dossiers")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Colonne `anticipation_jours` non disponible.")

# --- Tableau récapitulatif ---
st.markdown("---")
st.markdown("#### Aperçu des données")
st.dataframe(df.head(20), use_container_width=True, hide_index=True)
