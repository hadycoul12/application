"""
Page Impact Business — Simulation du CA sauvé par la rétention ciblée.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils.auth import (
    check_rgpd_consent,
    check_authentication,
    log_action,
    get_current_user,
)
from utils.predict import SEUIL

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("💰 Impact Business")
st.markdown(
    "Simulez le chiffre d'affaires sauvé en ciblant les dossiers à risque "
    "identifiés par le modèle."
)
st.markdown("---")

log_action(get_current_user(), "Accès page Impact Business")

# ---------------------------------------------------------------------------
# Paramètres de simulation (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Paramètres de simulation")

nb_dossiers_total = st.sidebar.number_input(
    "Nb de dossiers total (portefeuille annuel)",
    min_value=1000,
    max_value=1_000_000,
    value=310_000,
    step=10_000,
    help="Nombre total de réservations sur la période.",
)

taux_annulation = st.sidebar.slider(
    "Taux d'annulation (%)",
    min_value=1.0,
    max_value=30.0,
    value=7.27,
    step=0.1,
    help="Taux d'annulation observé dans les données.",
)

panier_moyen = st.sidebar.number_input(
    "Panier moyen (€ TTC)",
    min_value=100,
    max_value=5000,
    value=650,
    step=50,
    help="Chiffre d'affaires moyen par dossier.",
)

taux_retention = st.sidebar.slider(
    "Taux de rétention estimé (%)",
    min_value=1,
    max_value=50,
    value=15,
    step=1,
    help="Pourcentage de clients à risque qu'une action de rétention peut convaincre de maintenir.",
)

cout_action = st.sidebar.number_input(
    "Coût unitaire de l'action (€)",
    min_value=0,
    max_value=100,
    value=5,
    step=1,
    help="Coût d'un email/SMS/appel de rétention par dossier ciblé.",
)

# ---------------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------------
nb_annulations = int(nb_dossiers_total * taux_annulation / 100)
ca_perdu_total = nb_annulations * panier_moyen

# Scénario 1 : Sans modèle (aucune action)
ca_sauve_sans_modele = 0

# Scénario 2 : Modèle seuil 0.50 (rappel 62.7%, précision 16.8%)
rappel_050 = 0.627
precision_050 = 0.168
alertes_050 = int(nb_annulations * rappel_050 / precision_050)
annul_captees_050 = int(nb_annulations * rappel_050)
retenues_050 = int(annul_captees_050 * taux_retention / 100)
ca_sauve_050 = retenues_050 * panier_moyen
cout_total_050 = alertes_050 * cout_action
benefice_net_050 = ca_sauve_050 - cout_total_050

# Scénario 3 : Modèle seuil 0.30 (rappel 86.6%, précision ~11.4%)
rappel_030 = 0.866
precision_030 = 0.114
alertes_030 = int(nb_annulations * rappel_030 / precision_030)
annul_captees_030 = int(nb_annulations * rappel_030)
retenues_030 = int(annul_captees_030 * taux_retention / 100)
ca_sauve_030 = retenues_030 * panier_moyen
cout_total_030 = alertes_030 * cout_action
benefice_net_030 = ca_sauve_030 - cout_total_030

# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

# --- KPIs contextuels ---
st.markdown("### Contexte du portefeuille")
c1, c2, c3 = st.columns(3)
c1.metric("Dossiers total", f"{nb_dossiers_total:,}")
c2.metric("Annulations estimées", f"{nb_annulations:,}")
c3.metric("CA perdu (sans action)", f"{ca_perdu_total:,.0f} €")

st.markdown("---")

# --- Tableau comparatif des scénarios ---
st.markdown("### Comparaison des scénarios")

scenarios = pd.DataFrame(
    {
        "Scénario": [
            "Sans modèle",
            f"Seuil {SEUIL:.2f} (retenu)",
            "Seuil 0.30 (agressif)",
        ],
        "Alertes envoyées": [0, alertes_050, alertes_030],
        "Annulations captées": [0, annul_captees_050, annul_captees_030],
        "Clients retenus": [0, retenues_050, retenues_030],
        "CA sauvé (€)": [0, ca_sauve_050, ca_sauve_030],
        "Coût des actions (€)": [0, cout_total_050, cout_total_030],
        "Bénéfice net (€)": [0, benefice_net_050, benefice_net_030],
    }
)

st.dataframe(
    scenarios.style.format(
        {
            "Alertes envoyées": "{:,.0f}",
            "Annulations captées": "{:,.0f}",
            "Clients retenus": "{:,.0f}",
            "CA sauvé (€)": "{:,.0f}",
            "Coût des actions (€)": "{:,.0f}",
            "Bénéfice net (€)": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# --- Graphique comparatif ---
st.markdown("### Visualisation")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("#### CA sauvé vs Coût")
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(3)
    width = 0.35

    ca_values = [0, ca_sauve_050, ca_sauve_030]
    cout_values = [0, cout_total_050, cout_total_030]

    bars1 = ax.bar(x - width / 2, ca_values, width, label="CA sauvé", color="#2E8B57", alpha=0.85)
    bars2 = ax.bar(x + width / 2, cout_values, width, label="Coût actions", color="#E8593C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Sans modèle", f"Seuil {SEUIL}", "Seuil 0.30"], fontsize=9)
    ax.set_ylabel("Montant (€)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_g2:
    st.markdown("#### Bénéfice net par scénario")
    fig, ax = plt.subplots(figsize=(7, 5))

    benefices = [0, benefice_net_050, benefice_net_030]
    colors = ["#999999", "#2E8B57", "#3B8BD4"]

    bars = ax.bar(
        ["Sans modèle", f"Seuil {SEUIL}", "Seuil 0.30"],
        benefices,
        color=colors,
        alpha=0.85,
    )
    ax.bar_label(bars, fmt=lambda v: f"{v:,.0f} €", padding=4, fontsize=9)
    ax.set_ylabel("Bénéfice net (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- ROI ---
st.markdown("---")
st.markdown("### Retour sur investissement (ROI)")

if cout_total_050 > 0:
    roi_050 = (benefice_net_050 / cout_total_050) * 100
else:
    roi_050 = 0

if cout_total_030 > 0:
    roi_030 = (benefice_net_030 / cout_total_030) * 100
else:
    roi_030 = 0

r1, r2 = st.columns(2)
r1.metric(
    f"ROI seuil {SEUIL}",
    f"{roi_050:,.0f}%",
    help="(Bénéfice net / Coût actions) × 100",
)
r2.metric(
    "ROI seuil 0.30",
    f"{roi_030:,.0f}%",
    help="(Bénéfice net / Coût actions) × 100",
)

st.markdown("---")
st.caption(
    "💡 Ces simulations sont indicatives. Le taux de rétention réel dépend "
    "de la qualité de l'action commerciale (email, appel, offre de report). "
    "Un protocole A/B est recommandé pour mesurer l'efficacité réelle."
)
