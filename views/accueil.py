"""Accueil — présentation de l'outil, performances et diagnostic du modèle."""

import streamlit as st

import pandas as pd

from utils.model import base_rate, encoding_plan, field_specs, label, load_schema, seuil_alerte
from utils.ui import page_header, render_html, section

page_header(
    "Tableau de bord",
    "Scoring prédictif du risque d'annulation",
    "Identifiez les réservations à risque avant l'annulation et concentrez vos "
    "actions de rétention là où elles ont le plus d'impact.",
    show_logo=True,
)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    render_html("""
        <div class="tile">
            <div class="icon">📈</div>
            <h4>Comprendre</h4>
            <p>Explorez les tendances d'annulation par canal de vente, condition
            tarifaire, anticipation et destination.</p>
        </div>
        """)
with c2:
    render_html("""
        <div class="tile">
            <div class="icon">🎯</div>
            <h4>Prédire</h4>
            <p>Scorez un dossier ou un lot complet, et comprenez les facteurs de
            risque grâce aux explications SHAP.</p>
        </div>
        """)
with c3:
    render_html("""
        <div class="tile">
            <div class="icon">💰</div>
            <h4>Agir</h4>
            <p>Simulez le chiffre d'affaires sauvé par une campagne de rétention
            ciblée et arbitrez le seuil d'alerte.</p>
        </div>
        """)

section("Performances du modèle en production")

m1, m2, m3, m4 = st.columns(4)
m1.metric("PR-AUC", "0.263", "×3.6 vs aléatoire",
          help="Métrique principale sur données déséquilibrées (7,27 % de positifs).")
m2.metric("Rappel", "62.7 %", help="Part des annulations réelles détectées.")
m3.metric("Précision", "16.8 %", "×2.3 vs prévalence",
          help="Part des alertes correspondant à une annulation réelle.")
m4.metric("AUC-ROC", "0.764", help="Capacité de discrimination globale.")

st.caption(
    "XGBoost optimisé par RandomizedSearchCV (100 itérations, validation croisée 5-fold). "
    "Pondération native des classes, sans rééchantillonnage synthétique. "
    "Évaluation sur un holdout stratifié de 20 %."
)

section("Contexte")

a, b = st.columns([3, 2], gap="large")

with a:
    render_html("""
        <div class="card">
            <div class="card-title">Ce que le modèle sait faire</div>
            <div class="card-body">
                Le modèle capte les annulations <b>prévisibles</b> : clients bénéficiant
                d'une condition flexible, réservations très anticipées, certains canaux
                de distribution. Les variables structurelles portent l'essentiel du
                pouvoir prédictif.
                <br><br>
                Il reste aveugle aux annulations <b>« surprises »</b> — un client non
                assuré qui annule malgré la perte financière. Capter ces cas supposerait
                des signaux comportementaux aujourd'hui trop peu couverts.
            </div>
        </div>
        """)

with b:
    br = base_rate() * 100
    sa = seuil_alerte() * 100
    render_html(f"""
        <div class="card">
            <div class="card-title">Seuil de décision</div>
            <div class="card-body">
                Les probabilités sont <b>recalibrées</b> : un score affiché
                correspond à un risque réel. Un dossier est signalé « à risque
                élevé » au-delà de <b>{sa:.0f} %</b> — soit 2× le taux de base
                observé ({br:.1f} %).
                <br><br>
                Le seuil d'action optimal reste un arbitrage économique (coût
                d'une relance vs marge sauvée), analysé dans la page Impact business.
            </div>
        </div>
        """)

# --- Diagnostic technique ---------------------------------------------------
schema = load_schema()

with st.expander("Diagnostic technique du modèle", icon=":material/build:"):
    if schema is None:
        st.warning(
            "`data/schema.json` est absent. Lancez `python prepare_data.py "
            "<votre_dataset.csv>` pour le générer."
        )
    else:
        try:
            plan = encoding_plan()
        except Exception as e:
            st.error(f"Modèle illisible : {e}")
            st.stop()

        modes = {
            "pipeline": "Pipeline scikit-learn",
            "onehot": "One-hot (get_dummies)",
            "categorical": "Catégoriel natif XGBoost",
            "ordinal": "Encodage numérique",
            "passthrough": "Non déterminé",
        }

        d1, d2, d3 = st.columns(3)
        d1.metric("Structure détectée", modes.get(plan["mode"], plan["mode"]))
        d2.metric("Colonnes en entrée", len(plan["base_cols"]))
        d3.metric(
            "Étapes du pipeline" if plan["mode"] == "pipeline" else "Features du booster",
            plan.get("n_steps") if plan["mode"] == "pipeline" else len(plan["expected"]),
        )

        if plan["mode"] == "pipeline":
            st.success(
                "Le modèle est un **Pipeline scikit-learn** : le préprocessing "
                "(encodage, imputation) y est embarqué. L'application lui transmet les "
                "colonnes brutes et le pipeline applique lui-même les transformations — "
                "c'est le cas le plus fiable, aucune reconstitution d'encodage n'est "
                "nécessaire.",
                icon=":material/verified:",
            )
        else:
            st.caption(
                "La structure est déduite du modèle lui-même, non supposée. Si le mode "
                "détecté ne correspond pas à votre notebook, la prédiction échouera "
                "explicitement plutôt que de produire un score faux."
            )

        st.markdown("**Colonnes attendues en entrée**")

        specs = field_specs()
        libelles = {
            "schema": "Décrite dans schema.json",
            "data": "Absente du schéma, retrouvée dans les données",
            "absent": "Introuvable — champ neutre",
        }
        tableau = pd.DataFrame([
            {
                "Colonne": c,
                "Libellé": label(c),
                "Type": s["kind"],
                "Origine": libelles.get(s["source"], s["source"]),
            }
            for c, s in specs.items()
        ])

        manquantes = tableau[tableau["Origine"].str.startswith("Introuvable")]
        if len(manquantes):
            st.error(
                f"**{len(manquantes)} colonne(s) réclamée(s) par le modèle sont "
                "introuvables dans vos données.** Le formulaire propose un champ neutre, "
                "mais le score sera faussé. Régénérez `schema.json` à partir du jeu de "
                "données ayant réellement servi à l'entraînement.",
                icon=":material/error:",
            )

        st.dataframe(tableau, use_container_width=True, hide_index=True, height=300)

st.info(
    "Utilisez le menu latéral pour naviguer. Commencez par le **Dashboard** pour "
    "explorer le portefeuille, puis passez à la **Prédiction**.",
    icon=":material/lightbulb:",
)
