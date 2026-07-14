"""Accueil — présentation de l'outil, performances et diagnostic du modèle."""

import streamlit as st

from utils.model import encoding_plan, load_schema
from utils.ui import page_header, section

page_header(
    "Tableau de bord",
    "Scoring prédictif du risque d'annulation",
    "Identifiez les réservations à risque avant l'annulation et concentrez vos "
    "actions de rétention là où elles ont le plus d'impact.",
)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(
        """
        <div class="tile">
            <div class="icon">📈</div>
            <h4>Comprendre</h4>
            <p>Explorez les tendances d'annulation par canal de vente, condition
            tarifaire, anticipation et destination.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="tile">
            <div class="icon">🎯</div>
            <h4>Prédire</h4>
            <p>Scorez un dossier ou un lot complet, et comprenez les facteurs de
            risque grâce aux explications SHAP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="tile">
            <div class="icon">💰</div>
            <h4>Agir</h4>
            <p>Simulez le chiffre d'affaires sauvé par une campagne de rétention
            ciblée et arbitrez le seuil d'alerte.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Seuil de décision</div>
            <div class="card-body">
                Le seuil retenu est <b>0.50</b> : il maximise le F1 sous contrainte
                d'un rappel supérieur à 60 %.
                <br><br>
                L'abaisser augmente le rappel mais multiplie les fausses alertes —
                à 0.30, on capte 87 % des annulations au prix de 2,4× plus d'alertes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.code(", ".join(plan["base_cols"]), language=None)

st.info(
    "Utilisez le menu latéral pour naviguer. Commencez par le **Dashboard** pour "
    "explorer le portefeuille, puis passez à la **Prédiction**.",
    icon=":material/lightbulb:",
)
