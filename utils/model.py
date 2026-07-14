"""
Chargement du modèle et des données, découverte du schéma, alignement des
features et prédiction.

Point clé : aucune modalité n'est codée en dur. Les libellés (« CE Global »,
« Avt Hiver ») proviennent de `data/schema.json`, lui-même produit à partir
du dataset réel par `prepare_data.py`.

L'encodage attendu par le modèle est déduit du modèle lui-même
(`get_booster().feature_names` et `.feature_types`), ce qui couvre les trois
cas de figure : one-hot, encodage numérique, ou catégoriel natif XGBoost.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "xgb_optimise.joblib"
SCHEMA_PATH = ROOT / "data" / "schema.json"

DATA_CANDIDATES = [
    ROOT / "data" / "dataset_full.parquet",     # local — démo de soutenance
    ROOT / "data" / "sample_dataset.parquet",   # public — application déployée
    ROOT / "data" / "sample_dataset.csv",       # repli
]

TARGET = "y_annulation"
SEUIL = 0.50
SEUIL_MOYEN = 0.30

# Colonnes présentes dans les données mais jamais utilisées comme features
TECHNIQUES = {"dossier_cle", "id_dossier", "index"}

# Libellés lisibles. Toute colonne absente d'ici est simplement embellie.
LABELS = {
    "canal": "Canal de vente",
    "est_assure_annulation": "Assurance annulation (Flex)",
    "cond_annulation": "Condition d'annulation",
    "anticipation_jours": "Anticipation",
    "anticipation_tranche": "Tranche d'anticipation",
    "duree_sejour": "Durée du séjour",
    "nb_produits_total": "Nombre de produits",
    "a_prestations": "Prestations additionnelles",
    "device_resa": "Support de réservation",
    "nb_dossiers_anterieurs": "Dossiers antérieurs du client",
    "est_solo": "Réservation solo",
    "region_destination": "Région de destination",
    "theme_station": "Thème de la station",
    "periode_depart": "Période de départ",
    "groupe_fournisseur": "Groupe fournisseur",
    "est_dans_crm": "Présent dans le CRM",
    "nb_campagnes_recues": "Campagnes email reçues",
    "recence_email_jours": "Récence du dernier email",
    "assure_x_anticip": "Interaction assurance × anticipation",
}

# Regroupement des champs dans le formulaire de prédiction
GROUPES = {
    "Réservation": [
        "canal", "cond_annulation", "est_assure_annulation", "anticipation_jours",
        "anticipation_tranche", "duree_sejour", "nb_produits_total",
        "a_prestations", "device_resa",
    ],
    "Client": ["nb_dossiers_anterieurs", "est_solo"],
    "Destination": ["region_destination", "theme_station", "periode_depart",
                    "groupe_fournisseur"],
    "Engagement email": ["est_dans_crm", "nb_campagnes_recues", "recence_email_jours"],
}

UNITES = {
    "anticipation_jours": "jours",
    "duree_sejour": "nuits",
    "recence_email_jours": "jours",
}


def label(col: str) -> str:
    """Libellé lisible d'une colonne."""
    if col in LABELS:
        return LABELS[col]
    return col.replace("_", " ").capitalize()


# ===========================================================================
# Chargement
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"**Modèle introuvable.** Placez `xgb_optimise.joblib` dans `models/`.\n\n"
            f"Chemin attendu : `{MODEL_PATH}`"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Chargement du portefeuille…")
def load_data() -> tuple[pd.DataFrame | None, str | None]:
    """Charge le jeu de données le plus complet disponible."""
    for p in DATA_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        src = ("Jeu de données complet" if p.name == "dataset_full.parquet"
               else "Échantillon stratifié")
        return df, src
    return None, None


@st.cache_data(show_spinner=False)
def load_schema() -> dict | None:
    """Charge le schéma découvert dans les données."""
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return None


# ===========================================================================
# Détection de la structure du modèle
# ===========================================================================
# Quatre cas sont pris en charge :
#
#   pipeline     Le .joblib est un Pipeline scikit-learn — le préprocessing
#                (encodage, imputation, scaling) est embarqué. C'est le cas
#                le plus robuste : on lui passe les colonnes brutes.
#   onehot       XGBClassifier nu, entraîné sur des colonnes get_dummies.
#   categorical  XGBClassifier nu, enable_categorical=True.
#   ordinal      XGBClassifier nu, catégories encodées en entiers.
# ===========================================================================

def _is_pipeline(m) -> bool:
    return hasattr(m, "steps") and hasattr(m, "named_steps")


def _final_estimator(m):
    """Renvoie l'estimateur terminal (le XGBClassifier)."""
    return m.steps[-1][1] if _is_pipeline(m) else m


def _preprocessor(m):
    """Renvoie la partie préprocessing d'un Pipeline, ou None."""
    return m[:-1] if _is_pipeline(m) and len(m.steps) > 1 else None


def _names_in(obj) -> list[str]:
    """Colonnes d'entrée mémorisées par un estimateur sklearn, ou []."""
    if obj is None:
        return []
    noms = getattr(obj, "feature_names_in_", None)
    if noms is None:
        return []
    return [str(n) for n in noms]


def _booster_signature(m) -> dict:
    """Noms et types de features vus par le booster XGBoost."""
    est = _final_estimator(m)
    if not hasattr(est, "get_booster"):
        return {"names": [], "types": []}
    b = est.get_booster()
    return {"names": list(b.feature_names or []), "types": list(b.feature_types or [])}


@st.cache_data(show_spinner=False)
def encoding_plan() -> dict:
    """
    Déduit du modèle la façon de lui présenter les données.

    Retourne :
      mode        'pipeline' | 'onehot' | 'categorical' | 'ordinal'
      expected    colonnes vues par le booster (pour SHAP)
      base_cols   colonnes brutes à fournir en entrée (pour le formulaire)
      cat_cols    colonnes catégorielles concernées
      maps        {colonne: {modalité: code}} — mode 'ordinal' uniquement
      n_steps     nombre d'étapes du pipeline (0 si modèle nu)
    """
    m = load_model()
    schema = load_schema()
    cols = (schema or {}).get("columns", {})
    cat_cols = [c for c, s in cols.items() if s["kind"] == "categorical"]

    sig = _booster_signature(m)
    expected = sig["names"]
    types = sig["types"]

    # --- Cas 1 : Pipeline scikit-learn --------------------------------------
    if _is_pipeline(m):
        # sklearn mémorise les colonnes d'entrée quand le fit a porté sur un
        # DataFrame. `feature_names_in_` est un tableau numpy : on le convertit
        # explicitement, sans test de vérité (ambigu sur un tableau).
        entree = _names_in(m)
        if not entree:
            entree = _names_in(_preprocessor(m))
        if not entree:
            entree = list(cols)

        return {
            "mode": "pipeline",
            "expected": expected,
            "base_cols": entree,
            "cat_cols": [c for c in cat_cols if c in entree],
            "maps": {},
            "n_steps": len(m.steps),
        }

    if not expected:
        return {"mode": "passthrough", "expected": [], "base_cols": list(cols),
                "cat_cols": [], "maps": {}, "n_steps": 0}

    # --- Cas 2 : one-hot -----------------------------------------------------
    inconnues = [e for e in expected if e not in cols]
    prefixes = {c for c in cat_cols for e in inconnues if e.startswith(f"{c}_")}
    if prefixes and len(inconnues) > len(cat_cols):
        base = [c for c in cols if c in expected or c in prefixes]
        return {"mode": "onehot", "expected": expected, "base_cols": base,
                "cat_cols": sorted(prefixes), "maps": {}, "n_steps": 0}

    # --- Cas 3 : catégoriel natif XGBoost ------------------------------------
    if "c" in types:
        return {"mode": "categorical", "expected": expected, "base_cols": expected,
                "cat_cols": [c for c in cat_cols if c in expected],
                "maps": {}, "n_steps": 0}

    # --- Cas 4 : encodage numérique ------------------------------------------
    # LabelEncoder et .cat.codes attribuent les codes par ordre alphabétique ;
    # le schéma stocke les modalités triées, l'index reconstitue donc le code.
    maps = {c: {v: i for i, v in enumerate(cols[c]["values"])}
            for c in cat_cols if c in expected}
    return {"mode": "ordinal", "expected": expected, "base_cols": expected,
            "cat_cols": list(maps), "maps": maps, "n_steps": 0}


def align(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare un DataFrame pour `predict_proba`.

    En mode 'pipeline', on renvoie les colonnes BRUTES : c'est le pipeline
    lui-même qui applique l'encodage. Dans les autres modes, on reproduit
    l'encodage de l'entraînement.
    """
    plan = encoding_plan()
    mode = plan["mode"]

    if mode == "pipeline":
        attendues = plan["base_cols"]
        manquantes = [c for c in attendues if c not in df.columns]
        if manquantes:
            raise ValueError("Colonnes manquantes : "
                             + ", ".join(f"`{c}`" for c in manquantes))
        return df[attendues]

    expected = plan["expected"]
    if not expected:
        return df

    X = df.copy()

    if mode == "onehot":
        X = pd.get_dummies(X, columns=[c for c in plan["cat_cols"] if c in X.columns])
        # XGBoost refuse < > [ ] dans les noms : get_dummies est donc toujours
        # suivi d'un assainissement. Sans lui, « NoFlex <J30 » serait perdue.
        X.columns = [re.sub(r"[\[\]<>]", "", str(c)) for c in X.columns]
        return X.reindex(columns=expected, fill_value=0).astype(float)

    if mode == "categorical":
        for c in plan["cat_cols"]:
            if c in X.columns:
                X[c] = X[c].astype("category")
        manquantes = [c for c in expected if c not in X.columns]
        if manquantes:
            raise ValueError("Colonnes manquantes : "
                             + ", ".join(f"`{c}`" for c in manquantes))
        return X[expected]

    # ordinal
    for c, mp in plan["maps"].items():
        if c in X.columns:
            X[c] = X[c].astype(str).map(mp)
    manquantes = [c for c in expected if c not in X.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : "
                         + ", ".join(f"`{c}`" for c in manquantes))
    return X[expected].apply(pd.to_numeric, errors="coerce")


def transformed_frame(values: dict) -> tuple[pd.DataFrame, object, list[str]]:
    """
    Matrice de features telle que la voit le BOOSTER, l'estimateur terminal,
    et les noms de colonnes d'origine.

    SHAP travaille sur l'arbre, pas sur le pipeline : il faut donc appliquer
    le préprocessing soi-même. XGBoost refusant les caractères < > [ ] dans les
    noms de features, les colonnes sont assainies pour le calcul — les noms
    d'origine sont renvoyés à part, pour l'affichage.
    """
    m = load_model()
    plan = encoding_plan()
    X = align(pd.DataFrame([values]))

    if plan["mode"] != "pipeline":
        noms = [str(c) for c in X.columns]
        return X, m, noms

    pre = _preprocessor(m)
    est = _final_estimator(m)

    if pre is None:
        noms = [str(c) for c in X.columns]
        return X, est, noms

    Xt = pre.transform(X)

    try:
        noms = [str(n) for n in pre.get_feature_names_out()]
    except Exception:
        noms = plan["expected"] or [f"f{i}" for i in range(_ncols(Xt))]

    if hasattr(Xt, "toarray"):          # matrice creuse
        Xt = Xt.toarray()
    if isinstance(Xt, pd.DataFrame):
        Xt = Xt.to_numpy()

    noms = noms[: Xt.shape[1]]
    surs = [re.sub(r"[\[\]<>]", "", n) for n in noms]   # noms acceptés par XGBoost

    return pd.DataFrame(Xt, columns=surs), est, noms


def _ncols(a) -> int:
    return a.shape[1] if hasattr(a, "shape") else 0


# ===========================================================================
# Prédiction
# ===========================================================================
def risk_level(p: float) -> str:
    if p >= SEUIL:
        return "Élevé"
    if p >= SEUIL_MOYEN:
        return "Moyen"
    return "Faible"


def risk_css(level: str) -> str:
    return {"Élevé": "risk-high", "Moyen": "risk-mid", "Faible": "risk-low"}[level]


def predict_one(values: dict) -> float:
    """Probabilité d'annulation d'un dossier unique."""
    X = align(pd.DataFrame([values]))
    return float(load_model().predict_proba(X)[0, 1])


def predict_many(df: pd.DataFrame) -> pd.DataFrame:
    """Scoring d'un lot. Ajoute proba_annulation et risque."""
    X = align(df)
    out = df.copy()
    out["proba_annulation"] = load_model().predict_proba(X)[:, 1]
    out["risque"] = out["proba_annulation"].map(risk_level)
    return out.sort_values("proba_annulation", ascending=False)
