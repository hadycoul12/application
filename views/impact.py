"""Impact business, simulation du chiffre d'affaires sauvé."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.auth import current_user, log
from utils.model import base_rate, portfolio_scores, seuil_alerte
from utils.ui import page_header, section

page_header(
    "Décision",
    "Impact business",
    "Estimez le chiffre d'affaires préservé par une campagne de rétention ciblée "
    "et arbitrez le seuil d'alerte selon vos capacités opérationnelles.",
)

log(current_user(), "Consultation de l'impact business")

st.warning(
    "**Simulateur, résultats illustratifs.** Les montants affichés dépendent des "
    "hypothèses que vous saisissez (panier moyen, coût d'action, taux de rétention) : "
    "ce ne sont pas des gains réels mesurés. Faute de données de coûts Maeva, cette "
    "page opérationnalise le cadre de décision à titre exploratoire.",
    icon=":material/info:",
)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Hypothèses")

volume = st.sidebar.number_input(
    "Dossiers sur la période", 1_000, 2_000_000, 310_000, 10_000,
    help="Volume total de réservations sur l'horizon simulé.",
)
taux_annul = st.sidebar.slider(
    "Taux d'annulation (%)", 1.0, 30.0, 7.27, 0.01,
    help="Taux observé dans les données historiques.",
)
panier = st.sidebar.number_input(
    "Panier moyen (€)", 100, 5_000, 650, 50,
    help="Chiffre d'affaires moyen par dossier.",
)
retention = st.sidebar.slider(
    "Taux de rétention de l'action (%)", 1, 50, 15,
    help="Part des clients alertés qu'une action commerciale parvient à retenir.",
)
cout = st.sidebar.number_input(
    "Coût unitaire de l'action (€)", 0.0, 100.0, 5.0, 0.5,
    help="Coût d'un contact de rétention (email, SMS, appel).",
)

# ---------------------------------------------------------------------------
# Courbe seuil / rappel / précision, calculée EN DIRECT sur le portefeuille
# chargé, avec les probabilités CALIBRÉES et la vérité terrain (aucune valeur
# figée, échelle cohérente avec le reste de l'application).
# ---------------------------------------------------------------------------
proba_pf, y_pf = portfolio_scores()
s_alerte = round(seuil_alerte(), 3)


def _perf(seuil, proba, y):
    pred = proba >= seuil
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    rappel = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return rappel, precision


if proba_pf is not None and y_pf is not None:
    # Seuils étalés sur l'échelle calibrée, seuil d'alerte inclus.
    hauts = np.quantile(proba_pf, [0.80, 0.90, 0.95, 0.975])
    seuils = sorted({round(base_rate(), 3), s_alerte,
                     *(round(float(h), 3) for h in hauts)})
    COURBE = [(s, *_perf(s, proba_pf, y_pf)) for s in seuils]
else:
    # Repli : la vérité terrain n'est pas disponible dans le jeu chargé.
    COURBE = [(s_alerte, 0.63, 0.17)]

annulations = int(volume * taux_annul / 100)
ca_expose = annulations * panier


def simule(seuil, rappel, precision):
    captees = int(annulations * rappel)
    alertes = int(captees / precision) if precision > 0 else 0
    retenus = int(captees * retention / 100)
    ca_sauve = retenus * panier
    depense = alertes * cout
    net = ca_sauve - depense
    roi = (net / depense * 100) if depense > 0 else 0.0
    return dict(
        seuil=seuil, rappel=rappel, precision=precision,
        alertes=alertes, captees=captees, retenus=retenus,
        ca_sauve=ca_sauve, depense=depense, net=net, roi=roi,
    )


sims = [simule(*c) for c in COURBE]
best = max(sims, key=lambda s: s["net"])
ref = min(sims, key=lambda s: abs(s["seuil"] - s_alerte))

# ---------------------------------------------------------------------------
# Contexte
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dossiers", f"{volume:,}".replace(",", " "))
c2.metric("Annulations attendues", f"{annulations:,}".replace(",", " "))
c3.metric("CA exposé", f"{ca_expose/1e6:.2f} M€",
          help="Chiffre d'affaires perdu si aucune action n'est menée.")
c4.metric(f"Bénéfice net (seuil d'alerte {s_alerte:.2f})",
          f"{ref['net']/1e3:,.0f} k€".replace(",", " "),
          f"ROI {ref['roi']:.0f} %")

# ---------------------------------------------------------------------------
# Arbitrage du seuil
# ---------------------------------------------------------------------------
section("Arbitrage du seuil d'alerte")

fig = go.Figure()

fig.add_trace(go.Bar(
    x=[f"{s['seuil']:.2f}" for s in sims],
    y=[s["ca_sauve"] for s in sims],
    name="CA sauvé",
    marker=dict(color="#86EFAC", line=dict(width=0)),
    hovertemplate="Seuil %{x}<br>CA sauvé : %{y:,.0f} €<extra></extra>",
))
fig.add_trace(go.Bar(
    x=[f"{s['seuil']:.2f}" for s in sims],
    y=[s["depense"] for s in sims],
    name="Coût des actions",
    marker=dict(color="#FCA5A5", line=dict(width=0)),
    hovertemplate="Seuil %{x}<br>Coût : %{y:,.0f} €<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[f"{s['seuil']:.2f}" for s in sims],
    y=[s["net"] for s in sims],
    name="Bénéfice net",
    mode="lines+markers",
    line=dict(color="#1A1D24", width=3, shape="spline"),
    marker=dict(size=9, color="#1A1D24", line=dict(width=2, color="white")),
    hovertemplate="Seuil %{x}<br>Bénéfice net : %{y:,.0f} €<extra></extra>",
))

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#5A6270"),
    margin=dict(l=10, r=10, t=40, b=10), height=380,
    barmode="group", bargap=0.35,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(title="Seuil d'alerte", showgrid=False),
    yaxis=dict(title="Montant (€)", showgrid=True, gridcolor="#F1F3F7",
               zeroline=True, zerolinecolor="#C9CED8", tickformat=",.0f"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if best["seuil"] != ref["seuil"]:
    st.info(
        f"Avec ces hypothèses, le **seuil {best['seuil']:.2f}** maximise le bénéfice net "
        f"({best['net']:,.0f} €), contre {ref['net']:,.0f} € au seuil d'alerte "
        f"{ref['seuil']:.2f}. Il génère toutefois {best['alertes']:,} alertes à traiter."
        .replace(",", " "),
        icon=":material/lightbulb:",
    )
else:
    st.success(
        f"Le seuil d'alerte {ref['seuil']:.2f} maximise le bénéfice net avec ces hypothèses.",
        icon=":material/check_circle:",
    )

# ---------------------------------------------------------------------------
# Tableau détaillé
# ---------------------------------------------------------------------------
section("Détail par scénario")

table = pd.DataFrame([
    {
        "Seuil": f"{s['seuil']:.2f}",
        "Rappel": f"{s['rappel']*100:.1f} %",
        "Précision": f"{s['precision']*100:.1f} %",
        "Alertes à traiter": s["alertes"],
        "Annulations captées": s["captees"],
        "Clients retenus": s["retenus"],
        "CA sauvé (€)": s["ca_sauve"],
        "Coût (€)": s["depense"],
        "Bénéfice net (€)": s["net"],
        "ROI": f"{s['roi']:.0f} %",
    }
    for s in sims
])

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Alertes à traiter": st.column_config.NumberColumn(format="%d"),
        "Annulations captées": st.column_config.NumberColumn(format="%d"),
        "Clients retenus": st.column_config.NumberColumn(format="%d"),
        "CA sauvé (€)": st.column_config.NumberColumn(format="%d"),
        "Coût (€)": st.column_config.NumberColumn(format="%d"),
        "Bénéfice net (€)": st.column_config.NumberColumn(format="%d"),
    },
)

st.caption(
    "Les couples rappel/précision sont calculés en direct sur le portefeuille chargé, "
    "à partir des probabilités calibrées. Le taux de rétention reste une hypothèse "
    "commerciale : seul un protocole A/B (groupe traité vs groupe témoin) permet de "
    "le mesurer réellement."
)
