"""Scoring par lot, import CSV, scoring en masse, export."""

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.auth import current_user, log
from utils.model import (
    seuil_alerte, derive, encoding_plan, field_specs, label, load_schema, predict_many,
)
from utils.ui import page_header, section

HISTORY = Path(__file__).parent.parent / "data" / "predictions_history.csv"

page_header(
    "Scoring",
    "Scoring par lot",
    "Importez un fichier CSV de réservations pour scorer l'ensemble du lot et "
    "exporter la liste priorisée des dossiers à risque.",
)

schema = load_schema()
if schema is None:
    st.warning(
        "Le fichier `data/schema.json` est absent. Lancez `python prepare_data.py "
        "<votre_dataset.csv>` pour le générer.",
        icon=":material/warning:",
    )
    st.stop()

plan = encoding_plan()
# Les colonnes exigées viennent du MODÈLE, pas du schéma : c'est lui qui
# fait foi. Une colonne absente du schéma reste obligatoire dans le CSV.
SPECS = field_specs()
requises = list(SPECS)

with st.expander("Format de fichier attendu", icon=":material/description:"):
    st.markdown(
        "Le fichier doit contenir les colonnes suivantes, avec **les mêmes modalités "
        "que le dataset d'entraînement**. Les colonnes supplémentaires (par exemple "
        "`dossier_cle`) sont conservées à l'export mais ignorées par le modèle."
    )
    st.dataframe(
        pd.DataFrame({
            "Colonne": requises,
            "Libellé": [label(c) for c in requises],
            "Type": [SPECS[c]["kind"] for c in requises],
        }),
        use_container_width=True, hide_index=True, height=280,
    )
    st.caption(
        "`assure_x_anticip` est recalculé automatiquement s'il est absent, à partir "
        "de `est_assure_annulation` et `anticipation_jours`."
    )

up = st.file_uploader(
    "Fichier CSV", type=["csv"],
    help="Encodage UTF-8. Séparateur virgule ou point-virgule (détecté automatiquement).",
)

if up is None:
    st.stop()

try:
    brut = up.getvalue().decode("utf-8-sig")
    sep = ";" if brut[:4000].count(";") > brut[:4000].count(",") else ","
    df = pd.read_csv(io.StringIO(brut), sep=sep)
except Exception as e:
    st.error(f"Impossible de lire le fichier : {e}")
    st.stop()

st.success(
    f"**{len(df):,}** dossiers chargés · {len(df.columns)} colonnes".replace(",", " "),
    icon=":material/check_circle:",
)

# Variable dérivée recalculée si besoin
if ("assure_x_anticip" not in df.columns
        and {"est_assure_annulation", "anticipation_jours"} <= set(df.columns)):
    df["assure_x_anticip"] = df["est_assure_annulation"] * df["anticipation_jours"]
    st.caption("`assure_x_anticip` a été recalculée automatiquement.")

with st.expander("Aperçu des données importées"):
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

st.write("")
if not st.button("Scorer le lot", type="primary", use_container_width=True):
    st.stop()

with st.spinner(f"Scoring de {len(df):,} dossiers…".replace(",", " ")):
    try:
        out = predict_many(df)
    except ValueError as e:
        st.error(str(e), icon=":material/error:")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du scoring : {e}")
        st.stop()

n_haut = int((out["risque"] == "Élevé").sum())
n_moy = int((out["risque"] == "Moyen").sum())
n_bas = int((out["risque"] == "Faible").sum())

log(current_user(), f"Scoring batch · {len(out)} dossiers · {n_haut} à risque élevé")

section("Résultat du scoring")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dossiers scorés", f"{len(out):,}".replace(",", " "))
k2.metric("Risque élevé", f"{n_haut:,}".replace(",", " "),
          f"{n_haut / len(out) * 100:.1f} % du lot", delta_color="off")
k3.metric("Risque moyen", f"{n_moy:,}".replace(",", " "))
k4.metric("Risque faible", f"{n_bas:,}".replace(",", " "))

g1, g2 = st.columns([2, 1], gap="large")

with g1:
    st.markdown("**Distribution des scores**")
    fig = px.histogram(out, x="proba_annulation", nbins=40)
    fig.update_traces(marker=dict(color="#CBD5E1", line=dict(width=0)),
                      hovertemplate="Score : %{x:.2f}<br>Dossiers : %{y}<extra></extra>")
    _seuil = seuil_alerte()
    fig.add_vline(x=_seuil, line_dash="dash", line_color="#DC2626", line_width=2,
                  annotation_text=f"seuil {_seuil:.2f}", annotation_position="top",
                  annotation_font=dict(size=11, color="#DC2626"))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#566072"),
        margin=dict(l=10, r=10, t=30, b=10), height=300, bargap=.04,
        xaxis=dict(title="Probabilité d'annulation", showgrid=False),
        yaxis=dict(title="Dossiers", showgrid=True, gridcolor="#F1F3F7"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with g2:
    st.markdown("**Répartition**")
    counts = pd.DataFrame({"risque": ["Élevé", "Moyen", "Faible"],
                           "n": [n_haut, n_moy, n_bas]})
    fig = px.pie(counts, names="risque", values="n", hole=.62, color="risque",
                 color_discrete_map={"Élevé": "#DC2626", "Moyen": "#EA9A16",
                                     "Faible": "#16A34A"})
    fig.update_traces(textinfo="percent", textfont=dict(size=11, color="white"),
                      marker=dict(line=dict(color="white", width=2)),
                      hovertemplate="<b>%{label}</b><br>%{value:,} dossiers<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#566072"),
        margin=dict(l=0, r=0, t=30, b=0), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-.15, x=.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

section("Dossiers priorisés")

f1, _ = st.columns([1, 3])
with f1:
    sel = st.selectbox("Filtrer", ["Tous", "Élevé", "Moyen", "Faible"],
                       label_visibility="collapsed")

vue = out if sel == "Tous" else out[out["risque"] == sel]

st.dataframe(
    vue, use_container_width=True, hide_index=True, height=420,
    column_config={
        "proba_annulation": st.column_config.ProgressColumn(
            "Score", format="%.3f", min_value=0.0, max_value=1.0
        ),
    },
)
st.caption(f"{len(vue):,} dossiers affichés.".replace(",", " "))

st.download_button(
    "Télécharger les résultats (CSV)",
    data=out.to_csv(index=False, sep=";").encode("utf-8-sig"),
    file_name=f"scoring_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv", use_container_width=True, icon=":material/download:",
)

try:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "utilisateur": current_user(),
        "type": "batch",
        "proba_annulation": round(float(out["proba_annulation"].mean()), 4),
        "risque": f"{n_haut} élevés / {len(out)}",
    }]).to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False, encoding="utf-8")
except Exception:
    pass
