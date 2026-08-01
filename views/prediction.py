"""Prédiction individuelle — formulaire, score et explication SHAP."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.auth import current_user, log
from utils.explain import shap_drivers
from utils.model import (
    GROUPES, UNITES, default_for, derive, encoding_plan, field_specs,
    label, predict_one, risk_css, risk_level, seuil_alerte,
)
from utils.ui import page_header, render_html, section

HISTORY = Path(__file__).parent.parent / "data" / "predictions_history.csv"

page_header(
    "Scoring",
    "Prédiction individuelle",
    "Renseignez les caractéristiques d'une réservation pour obtenir son score de "
    "risque et comprendre les facteurs qui l'expliquent.",
)

try:
    plan = encoding_plan()
    SPECS = field_specs()
except Exception as e:
    st.error(f"Modèle illisible : {e}")
    st.stop()

if not SPECS:
    st.warning(
        "Impossible de déterminer les colonnes attendues par le modèle. "
        "Vérifiez que `models/xgb_optimise.joblib` et `data/schema.json` sont bien "
        "présents — ce dernier se génère avec `python prepare_data.py <dataset.csv>`.",
        icon=":material/warning:",
    )
    st.stop()

# --- Features calculables : on ne les demande pas à l'utilisateur -----------
CALCULEES = {"assure_x_anticip"}
sources_dispo = {"est_assure_annulation", "anticipation_jours"} <= set(SPECS)
calculees = CALCULEES & set(SPECS) if sources_dispo else set()

# --- Variables masquées : conservées pour le modèle, absentes du formulaire -
# Leur valeur par défaut est injectée en coulisses au moment du scoring.
masquees = {c: default_for(c, s) for c, s in SPECS.items()
            if s.get("hidden") and c not in calculees}

champs = [c for c in SPECS if c not in calculees and c not in masquees]

# --- Signalement des colonnes non retrouvées dans les données ---------------
orphelines = [c for c in champs if SPECS[c]["source"] == "absent"]
if orphelines:
    st.warning(
        "Ces colonnes sont réclamées par le modèle mais introuvables dans vos "
        "données : " + ", ".join(f"`{c}`" for c in orphelines) + ". "
        "Un champ neutre est proposé, mais le score sera peu fiable. "
        "Régénérez `schema.json` à partir du **jeu de données ayant servi à "
        "l'entraînement**, et non d'un export partiel.",
        icon=":material/warning:",
    )

# ===========================================================================
# Formulaire
# ===========================================================================
section("Caractéristiques du dossier")

groupes = {g: [c for c in cols if c in champs] for g, cols in GROUPES.items()}
restants = [c for c in champs if not any(c in v for v in groupes.values())]
if restants:
    groupes["Autres"] = restants
groupes = {g: c for g, c in groupes.items() if c}

values: dict = {}
onglets = st.tabs(list(groupes))

for onglet, (_, cols) in zip(onglets, groupes.items()):
    with onglet:
        colonnes = st.columns(min(3, len(cols)), gap="medium")

        for i, col in enumerate(cols):
            spec = SPECS[col]
            lib = label(col)
            if col in UNITES:
                lib = f"{lib} ({UNITES[col]})"
            aide = "Colonne introuvable dans les données" if spec["source"] == "absent" else None

            with colonnes[i % len(colonnes)]:
                if spec["kind"] == "categorical":
                    mods = spec["values"]
                    defaut = spec.get("default", mods[0])
                    values[col] = st.selectbox(
                        lib, options=mods,
                        index=mods.index(defaut) if defaut in mods else 0,
                        help=aide, key=f"f_{col}",
                    )

                elif spec["kind"] == "binary":
                    values[col] = st.selectbox(
                        lib, options=[0, 1],
                        format_func=lambda v: "Oui" if v == 1 else "Non",
                        index=int(spec.get("default", 0)),
                        help=aide, key=f"f_{col}",
                    )

                elif spec["kind"] == "text":
                    values[col] = st.text_input(
                        lib, value=str(spec.get("default", "")),
                        help=aide, key=f"f_{col}",
                    )

                else:  # numeric
                    if spec.get("is_int", True):
                        values[col] = st.number_input(
                            lib,
                            min_value=int(spec["min"]), max_value=int(spec["max"]),
                            value=int(spec["default"]), step=1,
                            help=aide, key=f"f_{col}",
                        )
                    else:
                        values[col] = st.number_input(
                            lib,
                            min_value=float(spec["min"]), max_value=float(spec["max"]),
                            value=float(spec["default"]),
                            help=aide, key=f"f_{col}",
                        )

# --- Valeurs masquées injectées puis features calculées ---------------------
values.update(masquees)
values = derive(values)

st.write("")
if not st.button("Scorer ce dossier", type="primary", use_container_width=True):
    st.stop()

# ===========================================================================
# Résultat
# ===========================================================================
with st.spinner("Calcul du score…"):
    try:
        proba = predict_one(values)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        n = len(plan["base_cols"]) or len(plan["expected"])
        st.info(
            f"Structure détectée : **{plan['mode']}** · {n} colonnes attendues · "
            f"{len(values)} colonnes fournies par le formulaire.",
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

render_html(f"""
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
    """)

couleur = {"Élevé": "#DC2626", "Moyen": "#EA9A16", "Faible": "#16A34A"}[level]
seuil = seuil_alerte() * 100
render_html(f"""
    <div class="gauge-track">
        <div class="gauge-threshold" style="left:{seuil}%;"></div>
        <div class="gauge-fill" style="left:{pct}%; color:{couleur};"></div>
    </div>
    <div class="gauge-scale">
        <span>0 %</span>
        <span style="color:#566072;font-weight:600;">seuil d'alerte {seuil:.0f} %</span>
        <span>100 %</span>
    </div>
    """)

# ===========================================================================
# SHAP
# ===========================================================================
section("Pourquoi ce score ?")

try:
    drivers = shap_drivers(values, top=8)
except Exception as e:
    st.warning(f"Explication indisponible : {e}")
    drivers = []

if drivers:
    gauche, droite = st.columns(2, gap="large")

    with gauche:
        st.markdown("**Principaux facteurs**")
        for d in drivers[:6]:
            fleche = "↑" if d["direction"] == "up" else "↓"
            render_html(f"""
                <div class="driver">
                    <div class="dir {d['direction']}">{fleche}</div>
                    <div class="name">{d['label']}</div>
                    <div class="val {d['direction']}">{d['shap']:+.3f}</div>
                </div>
                """)
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

# ===========================================================================
# Historique
# ===========================================================================
try:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "utilisateur": current_user(),
        "type": "individuelle",
        "proba_annulation": round(proba, 4),
        "risque": level,
        **values,
    }]).to_csv(HISTORY, mode="a", header=not HISTORY.exists(),
               index=False, encoding="utf-8")
    log(current_user(), f"Prédiction individuelle · proba={proba:.3f} · risque={level}")
    st.toast("Prédiction enregistrée", icon=":material/check_circle:")
except Exception:
    pass
