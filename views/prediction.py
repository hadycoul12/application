"""Prédiction individuelle — formulaire, score et explication SHAP."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.auth import current_user, log
from utils.explain import shap_drivers
from utils.model import (
    GROUPES, SEUIL, UNITES, encoding_plan, label, load_schema,
    predict_one, risk_css, risk_level,
)
from utils.ui import page_header, section

HISTORY = Path(__file__).parent.parent / "data" / "predictions_history.csv"

page_header(
    "Scoring",
    "Prédiction individuelle",
    "Renseignez les caractéristiques d'une réservation pour obtenir son score de "
    "risque et comprendre les facteurs qui l'expliquent.",
)

schema = load_schema()

if schema is None:
    st.warning(
        "Le fichier `data/schema.json` est absent. Lancez "
        "`python prepare_data.py <votre_dataset.csv>` pour le générer — c'est lui "
        "qui fournit les modalités réelles du formulaire.",
        icon=":material/warning:",
    )
    st.stop()

COLS = schema["columns"]

# Les features réellement attendues par le modèle
plan = encoding_plan()
attendues = set(plan["base_cols"]) or set(COLS)

# On ne saisit pas les variables dérivées : elles sont recalculées
DERIVEES = {"assure_x_anticip"}

champs = [c for c in COLS if c in attendues and c not in DERIVEES]

# ---------------------------------------------------------------------------
# Formulaire — regroupé, avec les modalités réelles du dataset
# ---------------------------------------------------------------------------
section("Caractéristiques du dossier")

# Répartition en groupes ; tout champ non listé va dans « Autres »
groupes = {g: [c for c in cols if c in champs] for g, cols in GROUPES.items()}
restants = [c for c in champs if not any(c in v for v in groupes.values())]
if restants:
    groupes["Autres"] = restants
groupes = {g: c for g, c in groupes.items() if c}

values: dict = {}
onglets = st.tabs(list(groupes))

for onglet, (nom, cols) in zip(onglets, groupes.items()):
    with onglet:
        colonnes = st.columns(min(3, len(cols)), gap="medium")

        for i, col in enumerate(cols):
            spec = COLS[col]
            with colonnes[i % len(colonnes)]:
                lib = label(col)

                if spec["kind"] == "categorical":
                    mods = spec["values"]
                    defaut = spec.get("default", mods[0])
                    values[col] = st.selectbox(
                        lib, options=mods,
                        index=mods.index(defaut) if defaut in mods else 0,
                        key=f"f_{col}",
                    )

                elif spec["kind"] == "binary":
                    values[col] = st.selectbox(
                        lib, options=[0, 1],
                        format_func=lambda v: "Oui" if v == 1 else "Non",
                        index=int(spec.get("default", 0)),
                        key=f"f_{col}",
                    )

                else:  # numeric
                    if spec.get("is_int", True):
                        values[col] = st.number_input(
                            f"{lib} ({UNITES[col]})" if col in UNITES else lib,
                            min_value=int(spec["min"]),
                            max_value=int(spec["max"]),
                            value=int(spec["default"]),
                            step=1,
                            key=f"f_{col}",
                        )
                    else:
                        values[col] = st.number_input(
                            f"{lib} ({UNITES[col]})" if col in UNITES else lib,
                            min_value=float(spec["min"]),
                            max_value=float(spec["max"]),
                            value=float(spec["default"]),
                            key=f"f_{col}",
                        )

# --- Variables dérivées recalculées ----------------------------------------
if "assure_x_anticip" in plan["base_cols"] or "assure_x_anticip" in COLS:
    a = values.get("est_assure_annulation", 0)
    b = values.get("anticipation_jours", 0)
    values["assure_x_anticip"] = a * b

st.write("")
if not st.button("Scorer ce dossier", type="primary", use_container_width=True):
    st.stop()

# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------
with st.spinner("Calcul du score…"):
    try:
        proba = predict_one(values)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.info(
            f"Encodage détecté : **{plan['mode']}** · "
            f"{len(plan['expected'])} features attendues par le modèle.",
            icon=":material/info:",
        )
        st.stop()

level = risk_level(proba)
pct = proba * 100

notes = {
    "Élevé": "Ce dossier dépasse le seuil d'alerte — une action de rétention est recommandée.",
    "Moyen": "Ce dossier est sous surveillance mais ne déclenche pas d'alerte.",
    "Faible": "Ce dossier présente un profil de maintien — aucune action requise.",
}

section("Résultat")

st.markdown(
    f"""
    <div class="risk-banner {risk_css(level)}">
        <div>
            <div class="lbl">Probabilité d'annulation</div>
            <div class="score">{pct:.1f} %</div>
        </div>
        <div style="text-align:right;">
            <div class="lbl">Verdict</div>
            <div class="verdict">Risque {level.lower()}</div>
            <div class="note">{notes[level]}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

couleur = {"Élevé": "#DC2626", "Moyen": "#EA9A16", "Faible": "#16A34A"}[level]
st.markdown(
    f"""
    <div class="gauge-track">
        <div class="gauge-threshold" style="left:{SEUIL * 100}%;"></div>
        <div class="gauge-fill" style="left:{pct}%; color:{couleur};"></div>
    </div>
    <div class="gauge-scale">
        <span>0 %</span>
        <span style="color:#566072;font-weight:600;">seuil d'alerte {SEUIL * 100:.0f} %</span>
        <span>100 %</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
section("Pourquoi ce score ?")

try:
    drivers = shap_drivers(values, top=8)
except Exception as e:
    st.warning(f"Explication indisponible : {e}")
    st.stop()

gauche, droite = st.columns(2, gap="large")

with gauche:
    st.markdown("**Principaux facteurs**")
    for d in drivers[:6]:
        fleche = "↑" if d["direction"] == "up" else "↓"
        st.markdown(
            f"""
            <div class="driver">
                <div class="dir {d['direction']}">{fleche}</div>
                <div class="name">{d['label']}</div>
                <div class="val {d['direction']}">{d['shap']:+.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Valeurs SHAP : contribution de chaque variable au score. "
        "Une valeur positive pousse vers l'annulation."
    )

with droite:
    st.markdown("**Décomposition**")
    dd = list(reversed(drivers))
    fig = go.Figure(go.Bar(
        x=[d["shap"] for d in dd],
        y=[d["label"] for d in dd],
        orientation="h",
        marker=dict(color=["#DC2626" if d["shap"] > 0 else "#16A34A" for d in dd],
                    line=dict(width=0)),
        text=[f"{d['shap']:+.3f}" for d in dd],
        textposition="outside",
        textfont=dict(size=10, color="#566072"),
        hovertemplate="<b>%{y}</b><br>Impact : %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#566072"),
        margin=dict(l=10, r=45, t=10, b=10), height=340, showlegend=False,
        xaxis=dict(title="", showgrid=True, gridcolor="#F1F3F7", zeroline=True,
                   zerolinecolor="#C9CED8", zerolinewidth=1.5),
        yaxis=dict(title="", showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Historique
# ---------------------------------------------------------------------------
try:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    ligne = {
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "utilisateur": current_user(),
        "type": "individuelle",
        "proba_annulation": round(proba, 4),
        "risque": level,
        **values,
    }
    pd.DataFrame([ligne]).to_csv(
        HISTORY, mode="a", header=not HISTORY.exists(), index=False, encoding="utf-8"
    )
    log(current_user(), f"Prédiction individuelle · proba={proba:.3f} · risque={level}")
    st.toast("Prédiction enregistrée", icon=":material/check_circle:")
except Exception:
    pass
