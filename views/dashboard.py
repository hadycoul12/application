"""Dashboard — exploration du portefeuille et analyse exploratoire."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.auth import current_user, log
from utils.model import TARGET, label, load_data, load_schema
from utils.ui import page_header, section

# --- Charte ----------------------------------------------------------------
ROUGE, GRIS, GRIS_PALE, ENCRE = "#E8593C", "#94A3B8", "#E2E8F0", "#1A1D24"
ECHELLE = ["#FDE4DC", "#F7A78F", "#E8593C", "#B33B22"]

BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#5A6270"),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
)
GRILLE = dict(showgrid=True, gridcolor="#F1F3F7", zeroline=False)
NUE = dict(showgrid=False, zeroline=False)
CFG = {"displayModeBar": False}


def montre(fig, h=320):
    fig.update_layout(**BASE, height=h)
    st.plotly_chart(fig, use_container_width=True, config=CFG)


def fr(n) -> str:
    return f"{n:,.0f}".replace(",", " ")


def taux_par(d: pd.DataFrame, col: str, min_vol: int = 1) -> pd.DataFrame:
    """Taux d'annulation et volume par modalité d'une colonne."""
    g = (
        d.groupby(col, observed=True)
        .agg(taux=(TARGET, "mean"), vol=(TARGET, "size"))
        .reset_index()
    )
    g["taux"] *= 100
    g[col] = g[col].astype(str)
    return g[g["vol"] >= min_vol]


# ===========================================================================
page_header(
    "Exploration",
    "Tableau de bord du portefeuille",
    "Analyse exploratoire des réservations et des facteurs associés à l'annulation.",
)

df, source = load_data()
schema = load_schema()

if df is None:
    st.warning(
        "Aucun jeu de données dans `data/`. Lancez "
        "`python prepare_data.py <votre_dataset.csv>`.",
        icon=":material/warning:",
    )
    st.stop()

if TARGET not in df.columns:
    st.error(f"La colonne cible `{TARGET}` est absente du jeu de données.")
    st.stop()

if schema is None:
    st.warning(
        "Le fichier `data/schema.json` est absent. Relancez `prepare_data.py` "
        "pour qu'il soit régénéré.",
        icon=":material/warning:",
    )
    st.stop()

COLS = schema["columns"]
CAT = [c for c, s in COLS.items() if s["kind"] == "categorical" and c in df.columns]
BIN = [c for c, s in COLS.items() if s["kind"] == "binary" and c in df.columns]
NUM = [c for c, s in COLS.items() if s["kind"] == "numeric" and c in df.columns]

log(current_user(), f"Dashboard ({source} · {len(df)} lignes)")

# ---------------------------------------------------------------------------
# Filtres — construits à partir des modalités réelles
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filtres")
d = df.copy()

# Deux filtres catégoriels : les colonnes ayant le plus faible nombre de modalités
filtrables = sorted(CAT, key=lambda c: len(COLS[c]["values"]))[:2]

for col in filtrables:
    mods = [m for m in COLS[col]["values"] if m in set(d[col].astype(str))]
    if len(mods) < 2:
        continue
    choix = st.sidebar.multiselect(label(col), options=mods, default=mods)
    if choix and len(choix) < len(mods):
        d = d[d[col].astype(str).isin(choix)]

if "est_assure_annulation" in BIN:
    flex = st.sidebar.radio("Assurance annulation", ["Toutes", "Assuré (Flex)", "Non assuré"])
    if flex == "Assuré (Flex)":
        d = d[d["est_assure_annulation"] == 1]
    elif flex == "Non assuré":
        d = d[d["est_assure_annulation"] == 0]

if d.empty:
    st.warning("Aucun dossier ne correspond aux filtres sélectionnés.")
    st.stop()

st.sidebar.caption(f"{fr(len(d))} dossiers sur {fr(len(df))}")
st.sidebar.caption(f"Source : {source.lower()}")

# --- Tranches d'anticipation (si la colonne brute existe) -------------------
ANTICIP = "anticipation_jours" if "anticipation_jours" in NUM else None
if ANTICIP:
    d = d.copy()
    d["_tranche"] = pd.cut(
        d[ANTICIP],
        bins=[-1, 7, 30, 60, 120, 10_000],
        labels=["≤ 7 j", "8–30 j", "31–60 j", "61–120 j", "> 120 j"],
    )

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
total, annul = len(d), int(d[TARGET].sum())
taux = annul / total * 100
ref = df[TARGET].mean() * 100
ecart = taux - ref

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dossiers", fr(total))
k2.metric("Annulations", fr(annul))
k3.metric("Taux d'annulation", f"{taux:.2f} %",
          f"{ecart:+.2f} pt vs global" if abs(ecart) > 0.01 else None,
          delta_color="inverse")
k4.metric("Dossiers maintenus", fr(total - annul))

tabs = st.tabs(["Vue d'ensemble", "Facteurs structurels", "Temporalité",
                "Engagement email", "Corrélations"])

# ===========================================================================
# 1. VUE D'ENSEMBLE
# ===========================================================================
with tabs[0]:
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown("**Déséquilibre de la cible**")
        rep = pd.DataFrame({"statut": ["Maintenu", "Annulé"], "n": [total - annul, annul]})
        fig = px.pie(rep, names="statut", values="n", hole=0.66, color="statut",
                     color_discrete_map={"Maintenu": GRIS_PALE, "Annulé": ROUGE})
        fig.update_traces(
            textinfo="percent", textfont=dict(size=12, color="white"),
            marker=dict(line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>%{value:,} dossiers<extra></extra>",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-.12,
                                      x=.5, xanchor="center"))
        montre(fig, 300)
        st.caption(
            f"Classe minoritaire à {taux:.2f} % — un déséquilibre marqué qui impose "
            "une pondération des classes et le PR-AUC comme métrique de référence."
        )

    with c2:
        if CAT:
            principal = max(CAT, key=lambda c: len(COLS[c]["values"]) if len(COLS[c]["values"]) <= 20 else 0)
            principal = "canal" if "canal" in CAT else principal
            st.markdown(f"**Taux d'annulation par {label(principal).lower()}**")

            g = taux_par(d, principal).sort_values("taux")
            fig = px.bar(
                g, x="taux", y=principal, orientation="h",
                text=g["taux"].map(lambda v: f"{v:.1f} %"),
                custom_data=["vol"], color="taux", color_continuous_scale=ECHELLE,
            )
            fig.update_traces(
                textposition="outside", textfont=dict(size=10, color="#5A6270"),
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Taux : %{x:.2f} %"
                              "<br>Volume : %{customdata[0]:,}<extra></extra>",
            )
            fig.add_vline(x=ref, line_dash="dot", line_color=ENCRE, opacity=.45,
                          annotation_text=f"moyenne {ref:.1f} %",
                          annotation_font=dict(size=10, color=ENCRE))
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(title="", ticksuffix=" %",
                           range=[0, g["taux"].max() * 1.28], **GRILLE),
                yaxis=dict(title="", **NUE),
            )
            montre(fig, max(300, 26 * len(g)))

    if CAT:
        section(f"Volume et risque par {label(principal).lower()}")

        g = taux_par(d, principal)
        g["annul"] = (g["taux"] / 100 * g["vol"]).round()

        fig = px.scatter(
            g, x="vol", y="taux", size="annul", text=principal,
            color="taux", color_continuous_scale=ECHELLE, size_max=52,
        )
        fig.update_traces(
            textposition="top center", textfont=dict(size=10, color="#5A6270"),
            marker=dict(line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>Volume : %{x:,}"
                          "<br>Taux : %{y:.2f} %<extra></extra>",
        )
        fig.add_hline(y=ref, line_dash="dot", line_color=ENCRE, opacity=.4)
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(title="Volume de dossiers", **GRILLE),
            yaxis=dict(title="Taux d'annulation", ticksuffix=" %", **GRILLE),
        )
        montre(fig, 400)
        st.caption(
            "La taille des bulles représente le nombre absolu d'annulations. Les modalités "
            "situées en haut à droite concentrent l'essentiel de la perte — gros volume "
            "**et** risque élevé. Ce sont les cibles prioritaires d'une action de rétention."
        )

# ===========================================================================
# 2. FACTEURS STRUCTURELS
# ===========================================================================
with tabs[1]:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        if "est_assure_annulation" in BIN:
            st.markdown("**Effet de l'assurance annulation**")
            g = taux_par(d, "est_assure_annulation")
            g["lib"] = g["est_assure_annulation"].map({"0": "Non assuré", "1": "Assuré (Flex)"})

            fig = px.bar(
                g, x="lib", y="taux", text=g["taux"].map(lambda v: f"{v:.2f} %"),
                custom_data=["vol"], color="lib",
                color_discrete_map={"Non assuré": GRIS, "Assuré (Flex)": ROUGE},
            )
            fig.update_traces(
                textposition="outside", textfont=dict(size=13, color="#5A6270"),
                marker_line_width=0, width=.5,
                hovertemplate="<b>%{x}</b><br>Taux : %{y:.2f} %"
                              "<br>Volume : %{customdata[0]:,}<extra></extra>",
            )
            fig.update_layout(
                showlegend=False,
                xaxis=dict(title="", **NUE),
                yaxis=dict(title="", ticksuffix=" %",
                           range=[0, g["taux"].max() * 1.35], **GRILLE),
            )
            montre(fig, 300)

            if len(g) == 2:
                t0 = g.loc[g.est_assure_annulation == "0", "taux"].iloc[0]
                t1 = g.loc[g.est_assure_annulation == "1", "taux"].iloc[0]
                st.caption(
                    f"Les clients assurés annulent **{t1 / max(t0, 1e-9):.1f}× plus** — "
                    "le coût nul de l'annulation lève le principal frein comportemental."
                )

    with c2:
        autres_bin = [b for b in BIN if b != "est_assure_annulation"]
        if autres_bin:
            st.markdown("**Autres variables binaires**")
            lignes = []
            for col in autres_bin:
                for v, sfx in [(0, "Non"), (1, "Oui")]:
                    sub = d[d[col] == v]
                    if len(sub):
                        lignes.append({
                            "var": label(col), "modalite": sfx,
                            "taux": sub[TARGET].mean() * 100, "vol": len(sub),
                        })
            g = pd.DataFrame(lignes)
            fig = px.bar(
                g, x="var", y="taux", color="modalite", barmode="group",
                text=g["taux"].map(lambda v: f"{v:.1f}"), custom_data=["vol"],
                color_discrete_map={"Non": GRIS_PALE, "Oui": ROUGE},
            )
            fig.update_traces(
                textposition="outside", textfont=dict(size=10, color="#5A6270"),
                marker_line_width=0,
                hovertemplate="<b>%{x} — %{fullData.name}</b><br>Taux : %{y:.2f} %"
                              "<br>Volume : %{customdata[0]:,}<extra></extra>",
            )
            fig.update_layout(
                legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis=dict(title="", **NUE),
                yaxis=dict(title="", ticksuffix=" %",
                           range=[0, g["taux"].max() * 1.32], **GRILLE),
            )
            montre(fig, 300)

    # --- Condition d'annulation (si présente comme catégorielle) ------------
    if "cond_annulation" in CAT:
        section("Condition d'annulation")
        g = taux_par(d, "cond_annulation").sort_values("taux")
        fig = px.bar(
            g, x="cond_annulation", y="taux",
            text=g["taux"].map(lambda v: f"{v:.2f} %"), custom_data=["vol"],
            color="taux", color_continuous_scale=ECHELLE,
        )
        fig.update_traces(
            textposition="outside", textfont=dict(size=11, color="#5A6270"),
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Taux : %{y:.2f} %"
                          "<br>Volume : %{customdata[0]:,}<extra></extra>",
        )
        fig.add_hline(y=ref, line_dash="dot", line_color=ENCRE, opacity=.4)
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(title="", **NUE),
            yaxis=dict(title="", ticksuffix=" %",
                       range=[0, g["taux"].max() * 1.3], **GRILLE),
        )
        montre(fig, 340)
        st.caption(
            "Le gradient des conditions tarifaires — du plus flexible au plus contraignant — "
            "est l'un des signaux structurels les plus nets du modèle."
        )

    # --- Interaction assurance x anticipation -------------------------------
    if ANTICIP and "est_assure_annulation" in BIN:
        section("Interaction assurance × anticipation")

        g = (
            d.groupby(["_tranche", "est_assure_annulation"], observed=True)
            .agg(taux=(TARGET, "mean"), vol=(TARGET, "size"))
            .reset_index()
        )
        g["taux"] *= 100
        g["lib"] = g["est_assure_annulation"].map({0: "Non assuré", 1: "Assuré (Flex)"})
        g["_tranche"] = g["_tranche"].astype(str)

        fig = px.line(
            g, x="_tranche", y="taux", color="lib", markers=True, custom_data=["vol"],
            color_discrete_map={"Non assuré": GRIS, "Assuré (Flex)": ROUGE},
        )
        fig.update_traces(
            line=dict(width=3, shape="spline"),
            marker=dict(size=10, line=dict(width=2, color="white")),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Taux : %{y:.2f} %"
                          "<br>Volume : %{customdata[0]:,}<extra></extra>",
        )
        fig.update_layout(
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(title="Délai d'anticipation", **NUE),
            yaxis=dict(title="Taux d'annulation", ticksuffix=" %", **GRILLE),
            hovermode="x unified",
        )
        montre(fig, 360)
        st.caption(
            "L'écart entre les deux courbes se creuse avec l'anticipation : c'est ce que "
            "capte la variable dérivée `assure_x_anticip`, l'un des principaux contributeurs "
            "du modèle selon les valeurs SHAP."
        )

# ===========================================================================
# 3. TEMPORALITÉ
# ===========================================================================
with tabs[2]:
    if ANTICIP:
        st.markdown("**Gradient d'anticipation**")
        g = (
            d.groupby("_tranche", observed=True)
            .agg(taux=(TARGET, "mean"), vol=(TARGET, "size"))
            .reset_index()
        )
        g["taux"] *= 100
        g["_tranche"] = g["_tranche"].astype(str)

        fig = go.Figure()
        fig.add_bar(x=g["_tranche"], y=g["vol"], name="Volume",
                    marker=dict(color=GRIS_PALE), yaxis="y",
                    hovertemplate="<b>%{x}</b><br>Volume : %{y:,}<extra></extra>")
        fig.add_scatter(
            x=g["_tranche"], y=g["taux"], name="Taux d'annulation",
            mode="lines+markers+text", text=g["taux"].map(lambda v: f"{v:.1f} %"),
            textposition="top center", textfont=dict(size=11, color=ROUGE),
            line=dict(color=ROUGE, width=3, shape="spline"),
            marker=dict(size=10, line=dict(width=2, color="white")), yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Taux : %{y:.2f} %<extra></extra>",
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(title="", **NUE),
            yaxis=dict(title="Volume", **GRILLE),
            yaxis2=dict(title="Taux", overlaying="y", side="right", ticksuffix=" %",
                        range=[0, g["taux"].max() * 1.45], **NUE),
        )
        montre(fig, 360)
        st.caption(
            "Plus la réservation est anticipée, plus le risque croît : le délai laisse le "
            "temps aux imprévus et aux changements d'avis. C'est le gradient le plus net "
            "de l'analyse exploratoire."
        )

    c1, c2 = st.columns(2, gap="large")

    # Saisonnalité — la colonne période peut être texte OU numérique
    with c1:
        per = "periode_depart" if "periode_depart" in df.columns else None
        if per:
            st.markdown(f"**{label(per)}**")
            g = taux_par(d, per).sort_values("taux")
            fig = px.bar(
                g, x="taux", y=per, orientation="h",
                text=g["taux"].map(lambda v: f"{v:.1f} %"), custom_data=["vol"],
                color="taux", color_continuous_scale=ECHELLE,
            )
            fig.update_traces(
                textposition="outside", textfont=dict(size=10, color="#5A6270"),
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Taux : %{x:.2f} %"
                              "<br>Volume : %{customdata[0]:,}<extra></extra>",
            )
            fig.add_vline(x=ref, line_dash="dot", line_color=ENCRE, opacity=.4)
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(title="", ticksuffix=" %",
                           range=[0, g["taux"].max() * 1.3], **GRILLE),
                yaxis=dict(title="", **NUE),
            )
            montre(fig, max(300, 30 * len(g)))

    with c2:
        if "duree_sejour" in df.columns:
            st.markdown("**Durée du séjour**")
            g = taux_par(d, "duree_sejour", min_vol=max(30, int(len(d) * .001)))
            g["duree_sejour"] = pd.to_numeric(g["duree_sejour"], errors="coerce")
            g = g.dropna(subset=["duree_sejour"])

            if len(g):
                fig = px.scatter(
                    g, x="duree_sejour", y="taux", size="vol",
                    color="taux", color_continuous_scale=ECHELLE, size_max=30,
                )
                fig.update_traces(
                    marker=dict(line=dict(width=1.5, color="white")),
                    hovertemplate="Durée : %{x} nuits<br>Taux : %{y:.2f} %<extra></extra>",
                )
                fig.add_hline(y=ref, line_dash="dot", line_color=ENCRE, opacity=.4)
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    xaxis=dict(title="Nuits", **GRILLE),
                    yaxis=dict(title="", ticksuffix=" %", **GRILLE),
                )
                montre(fig, 300)
                st.caption("La taille des points reflète le volume de dossiers.")

    if ANTICIP:
        section("Distribution du délai d'anticipation")
        cap = float(d[ANTICIP].quantile(.99))
        fig = go.Figure()
        for lab, val, col in [("Maintenu", 0, GRIS_PALE), ("Annulé", 1, ROUGE)]:
            fig.add_histogram(
                x=d.loc[d[TARGET] == val, ANTICIP].clip(upper=cap),
                name=lab, nbinsx=50, histnorm="probability density",
                marker=dict(color=col, line=dict(width=0)), opacity=.75,
                hovertemplate=f"<b>{lab}</b><br>%{{x:.0f}} j<extra></extra>",
            )
        fig.update_layout(
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(title="Anticipation (jours — tronquée au 99ᵉ centile)", **NUE),
            yaxis=dict(title="Densité", **GRILLE),
        )
        montre(fig, 330)
        st.caption(
            "Densités normalisées, pour rendre les deux classes comparables malgré le "
            "déséquilibre. La distribution des annulations est nettement décalée vers la droite."
        )

# ===========================================================================
# 4. ENGAGEMENT EMAIL
# ===========================================================================
with tabs[3]:
    if "est_dans_crm" not in df.columns:
        st.info("Les variables d'engagement email sont absentes de ce jeu de données.")
    else:
        couverture = d["est_dans_crm"].mean() * 100
        crm = d[d["est_dans_crm"] == 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Couverture CRM", f"{couverture:.1f} %",
                  help="Part des dossiers disposant d'un historique email exploitable.")
        c2.metric("Dossiers couverts", fr(len(crm)))
        c3.metric("Taux d'annulation (couverts)",
                  f"{crm[TARGET].mean() * 100:.2f} %" if len(crm) else "—",
                  f"{crm[TARGET].mean() * 100 - taux:+.2f} pt" if len(crm) else None,
                  delta_color="inverse")

        st.warning(
            f"Seuls **{couverture:.1f} %** des dossiers disposent d'un historique email. "
            "Sur les autres, les variables d'engagement valent zéro — ce qui dilue leur "
            "pouvoir prédictif dans le modèle global et justifie l'analyse Tier 2 menée "
            "sur le seul sous-ensemble CRM.",
            icon=":material/info:",
        )

        if len(crm) > 100:
            c1, c2 = st.columns(2, gap="large")

            for col_st, col, bins, libs, titre in [
                (c1, "nb_campagnes_recues", [-1, 0, 2, 5, 10, 1e9],
                 ["0", "1–2", "3–5", "6–10", "> 10"], "Campagnes reçues"),
                (c2, "recence_email_jours", [-1, 7, 30, 60, 90, 1e9],
                 ["≤ 7 j", "8–30 j", "31–60 j", "61–90 j", "> 90 j"], "Récence du dernier email"),
            ]:
                if col not in crm.columns:
                    continue
                with col_st:
                    st.markdown(f"**{titre}** *(sous-ensemble CRM)*")
                    tmp = crm.copy()
                    tmp["_b"] = pd.cut(tmp[col], bins=bins, labels=libs)
                    g = (
                        tmp.groupby("_b", observed=True)
                        .agg(taux=(TARGET, "mean"), vol=(TARGET, "size"))
                        .reset_index()
                    )
                    g = g[g["vol"] >= 20]
                    g["taux"] *= 100
                    g["_b"] = g["_b"].astype(str)

                    if len(g):
                        fig = px.bar(
                            g, x="_b", y="taux", custom_data=["vol"],
                            text=g["taux"].map(lambda v: f"{v:.1f} %"),
                            color="taux", color_continuous_scale=ECHELLE,
                        )
                        fig.update_traces(
                            textposition="outside", textfont=dict(size=10, color="#5A6270"),
                            marker_line_width=0,
                            hovertemplate="<b>%{x}</b><br>Taux : %{y:.2f} %"
                                          "<br>Volume : %{customdata[0]:,}<extra></extra>",
                        )
                        fig.update_layout(
                            showlegend=False, coloraxis_showscale=False,
                            xaxis=dict(title="", **NUE),
                            yaxis=dict(title="", ticksuffix=" %",
                                       range=[0, g["taux"].max() * 1.35], **GRILLE),
                        )
                        montre(fig, 310)

            st.caption(
                "Ces gradients restent faibles comparés à ceux des variables structurelles — "
                "cohérent avec le classement SHAP, où les variables email figurent en bas."
            )

# ===========================================================================
# 5. CORRÉLATIONS
# ===========================================================================
with tabs[4]:
    num = d[[c for c in NUM + BIN if c in d.columns] + [TARGET]].select_dtypes(
        include=[np.number]
    )

    if TARGET not in num.columns or num.shape[1] < 2:
        st.info("Pas assez de variables numériques pour calculer les corrélations.")
    else:
        st.markdown("**Corrélation de chaque variable avec la cible**")
        st.caption(
            "Seules les variables numériques et binaires figurent ici — le coefficient "
            "de Pearson n'a pas de sens sur une variable catégorielle non ordonnée."
        )

        corr = num.corr()[TARGET].drop(TARGET).dropna().sort_values()
        cdf = pd.DataFrame({"var": corr.index, "r": corr.values})
        cdf["lib"] = cdf["var"].map(label)

        fig = px.bar(
            cdf, x="r", y="lib", orientation="h",
            text=cdf["r"].map(lambda v: f"{v:+.3f}"), color="r",
            color_continuous_scale=["#3B82F6", "#E2E8F0", ROUGE],
            color_continuous_midpoint=0,
        )
        fig.update_traces(
            textposition="outside", textfont=dict(size=10, color="#5A6270"),
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>r = %{x:+.4f}<extra></extra>",
        )
        span = max(abs(cdf["r"]).max() * 1.45, .05)
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(title="Coefficient de Pearson", zeroline=True,
                       zerolinecolor="#C9CED8", zerolinewidth=1.5,
                       showgrid=True, gridcolor="#F1F3F7", range=[-span, span]),
            yaxis=dict(title="", **NUE),
        )
        montre(fig, max(320, 28 * len(cdf)))
        st.caption(
            "Les corrélations linéaires restent faibles en valeur absolue — attendu sur un "
            "problème déséquilibré aux effets non linéaires. C'est ce qui justifie le recours "
            "à un modèle d'ensemble comme XGBoost plutôt qu'à une régression logistique."
        )

        section("Matrice de corrélation")
        cols = list(num.columns)[:15]
        m = num[cols].corr()
        libs = [label(c) for c in m.columns]

        fig = go.Figure(go.Heatmap(
            z=m.values, x=libs, y=libs,
            colorscale=[[0, "#3B82F6"], [.5, "#F8FAFC"], [1, ROUGE]],
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>r = %{z:+.3f}<extra></extra>",
            colorbar=dict(thickness=12, len=.75, outlinewidth=0),
        ))
        fig.update_layout(
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), **NUE),
            yaxis=dict(tickfont=dict(size=10), **NUE),
        )
        montre(fig, 520)
        st.caption(
            "Une corrélation très élevée entre deux variables explicatives signalerait une "
            "redondance — attendu entre `assure_x_anticip` et ses composantes, la variable "
            "dérivée étant précisément conçue pour capter leur interaction."
        )

# ---------------------------------------------------------------------------
section("Données brutes")
with st.expander("Consulter un échantillon"):
    st.dataframe(d.drop(columns=["_tranche"], errors="ignore").head(100),
                 use_container_width=True, hide_index=True)
    st.caption(f"100 premières lignes sur {fr(len(d))} · source : {source.lower()}")
