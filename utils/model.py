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
CALIB_PATH = ROOT / "models" / "calibrateur.joblib"
SCHEMA_PATH = ROOT / "data" / "schema.json"

DATA_CANDIDATES = [
    ROOT / "data" / "dataset_full.parquet",     # local — démo de soutenance
    ROOT / "data" / "sample_dataset.parquet",   # public — application déployée
    ROOT / "data" / "sample_dataset.csv",       # repli
]

TARGET = "y_annulation"

# Seuils de risque RELATIFS au taux de base (jamais codés en dur — voir base_rate).
# Les probabilités sont recalibrées (isotonique, voir load_calibrator) : un score
# affiché reflète alors un vrai risque, ce qui rend cette échelle relative fiable.
#   « Élevé » = proba ≥ FACTEUR_ELEVE × taux de base
#   « Moyen » = proba ≥ taux de base
FACTEUR_ELEVE = 2.0

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
    "nb_bebe": "Nombre de bébés",
    "nb_bebes": "Nombre de bébés",
    "nb_mineur": "Nombre de mineurs",
    "nb_mineurs": "Nombre de mineurs",
    "nb_enfants": "Nombre d'enfants",
    "nb_adultes": "Nombre d'adultes",
    "dossier_nb_pax_total": "Nombre de voyageurs",
    "dossier_nb_pax_adultes": "Nombre d'adultes",
    "nb_pax_total": "Nombre de voyageurs",
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

# ---------------------------------------------------------------------------
# Variables retirées du FORMULAIRE de prédiction.
# Elles restent utilisées par le modèle : on ne les demande simplement pas au
# gestionnaire, car leur valeur est peu actionnable ou déjà connue du système.
# En coulisses, chacune reçoit une valeur par défaut (voir default_for()).
# ---------------------------------------------------------------------------
MASQUEES = {
    "est_assure_annulation",   # assurance annulation
    "assure_x_anticip",        # interaction dérivée de l'assurance
    "groupe_fournisseur",      # groupe fournisseur
    "type_produit",            # type de produit
    "type_produit_principal",
    "client_vip",              # client VIP
    "est_client_vip",
    "est_vip",
    "vip",
}

# Colonnes de comptage : forcées en entier même si le CSV les stocke en float
# (un nombre de bébés ou de mineurs ne peut pas valoir 1,4).
COMPTAGES = {
    "nb_bebe", "nb_bebes", "nb_mineur", "nb_mineurs", "nb_enfants",
    "nb_adultes", "nb_pax", "nb_pax_total", "nb_pax_adultes",
    "dossier_nb_pax_total", "dossier_nb_pax_adultes", "dossier_nb_pax",
    "nb_produits_total", "nb_dossiers_anterieurs", "nb_campagnes_recues",
}


def is_comptage(col: str) -> bool:
    """Vrai si la colonne est un comptage (doit rester entier)."""
    c = col.lower()
    return c in COMPTAGES or c.startswith("nb_") or "_nb_" in c


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


@st.cache_resource(show_spinner=False)
def load_calibrator():
    """
    Charge le calibrateur isotonique s'il existe, sinon None.

    Le calibrateur (produit par le notebook, section « Recalibration ») recale
    les probabilités brutes du modèle sur la réalité observée. Il est facultatif :
    si le fichier est absent, l'application fonctionne comme avant (scores bruts).
    """
    if not CALIB_PATH.exists():
        return None
    try:
        obj = joblib.load(CALIB_PATH)
        return obj.get("calibrateur") if isinstance(obj, dict) else obj
    except Exception:
        return None


def _calibrate(p):
    """Applique le calibrateur aux probabilités brutes, si disponible."""
    cal = load_calibrator()
    if cal is None:
        return np.asarray(p, dtype=float).ravel()
    return cal.predict(np.asarray(p, dtype=float).ravel())


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

    # Cas 1 : Pipeline scikit-learn
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

    # Cas 2 : one-hot
    inconnues = [e for e in expected if e not in cols]
    prefixes = {c for c in cat_cols for e in inconnues if e.startswith(f"{c}_")}
    if prefixes and len(inconnues) > len(cat_cols):
        base = [c for c in cols if c in expected or c in prefixes]
        return {"mode": "onehot", "expected": expected, "base_cols": base,
                "cat_cols": sorted(prefixes), "maps": {}, "n_steps": 0}

    # Cas 3 : catégoriel natif XGBoost
    if "c" in types:
        return {"mode": "categorical", "expected": expected, "base_cols": expected,
                "cat_cols": [c for c in cat_cols if c in expected],
                "maps": {}, "n_steps": 0}

    #  Cas 4 : encodage numérique 
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
@st.cache_data(show_spinner=False)
def base_rate() -> float:
    """
    Taux d'annulation de référence, lu dans les données — jamais codé en dur.

    C'est l'ancre des niveaux de risque : un dossier « moyen » dépasse ce taux,
    un dossier « élevé » en dépasse le double. Repli neutre (0.5) si les données
    sont indisponibles, ce qui désactive de fait l'échelle relative.
    """
    df, _ = load_data()
    if df is not None and TARGET in df.columns:
        r = float(df[TARGET].mean())
        if 0.0 < r < 1.0:
            return r
    return 0.5


def seuil_alerte() -> float:
    """Seuil d'alerte « risque élevé » = FACTEUR_ELEVE × taux de base."""
    return min(1.0, FACTEUR_ELEVE * base_rate())


def risk_level(p: float) -> str:
    br = base_rate()
    if p >= FACTEUR_ELEVE * br:
        return "Élevé"
    if p >= br:
        return "Moyen"
    return "Faible"


def risk_css(level: str) -> str:
    return {"Élevé": "risk-high", "Moyen": "risk-mid", "Faible": "risk-low"}[level]


def predict_one(values: dict) -> float:
    """Probabilité d'annulation calibrée d'un dossier unique."""
    X = align(pd.DataFrame([values]))
    raw = load_model().predict_proba(X)[0, 1]
    return float(_calibrate([raw])[0])


def predict_many(df: pd.DataFrame) -> pd.DataFrame:
    """Scoring d'un lot. Ajoute proba_annulation (calibrée) et risque."""
    X = align(df)
    out = df.copy()
    raw = load_model().predict_proba(X)[:, 1]
    out["proba_annulation"] = _calibrate(raw)
    out["risque"] = out["proba_annulation"].map(risk_level)
    return out.sort_values("proba_annulation", ascending=False)


@st.cache_data(show_spinner="Évaluation du portefeuille…")
def portfolio_scores():
    """
    Probabilités calibrées + vérité terrain sur le portefeuille chargé.

    Sert à construire, en direct, la courbe rappel/précision du modèle (page
    Impact) — plutôt que des chiffres figés. Retourne (proba, y) sous forme de
    tableaux numpy, ou (None, None) si les données ou la cible sont indisponibles.
    """
    df, _ = load_data()
    if df is None or TARGET not in df.columns:
        return None, None
    try:
        X = align(df)
    except Exception:
        return None, None
    proba = _calibrate(load_model().predict_proba(X)[:, 1])
    return np.asarray(proba, dtype=float), df[TARGET].to_numpy()


# ===========================================================================
# Spécification des champs du formulaire
# ===========================================================================
# Principe : c'est le MODÈLE qui dicte les champs, pas le schéma.
#
# Pour chaque colonne réclamée par le modèle, on cherche sa description dans
# `schema.json` ; si elle n'y figure pas (feature créée dans le notebook et
# absente du CSV exporté, par exemple), on l'infère directement du jeu de
# données chargé. En dernier recours, on fournit un champ neutre plutôt que
# d'échouer.
#
# Ce renversement garantit qu'aucune colonne attendue ne peut manquer à
# l'appel au moment du scoring.
# ===========================================================================

def _infer_spec(s: pd.Series, col: str = "") -> dict:
    """Déduit une spécification de saisie à partir d'une colonne de données."""
    if not pd.api.types.is_numeric_dtype(s):
        mods = sorted(str(v) for v in s.dropna().unique())
        if 0 < len(mods) <= 60:
            return {
                "kind": "categorical",
                "values": mods,
                "default": str(s.mode().iloc[0]) if len(s.mode()) else mods[0],
            }
        return {"kind": "text", "default": ""}

    vals = set(pd.unique(s.dropna()))
    if vals and vals <= {0, 1, 0.0, 1.0, True, False}:
        return {"kind": "binary", "default": int(s.mode().iloc[0]) if len(s.mode()) else 0}

    # Un comptage reste entier même si le CSV le stocke en float (1.0, 2.0…)
    entier = bool(pd.api.types.is_integer_dtype(s)) or is_comptage(col)
    mediane = float(np.nanmedian(s))
    return {
        "kind": "numeric",
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
        "default": round(mediane) if entier else mediane,
        "is_int": entier,
    }


@st.cache_data(show_spinner=False)
def field_specs() -> dict:
    """
    Spécification de saisie pour CHAQUE colonne réclamée par le modèle.

    Chaque entrée porte une clé `source`, par ordre de fiabilité :
      modele  modalités lues dans l'encodeur ajusté du pipeline (référence)
      schema  décrite dans schema.json
      data    inférée du jeu de données chargé
      absent  introuvable — champ neutre, score peu fiable
    """
    plan = encoding_plan()
    requises = plan["base_cols"] or list((load_schema() or {}).get("columns", {}))

    cols = (load_schema() or {}).get("columns", {})
    df, _ = load_data()
    vocab, numeriques = pipeline_vocab()

    specs: dict = {}
    for c in requises:
        # 1. Le pipeline fait foi : il connaît les modalités vues à l'entraînement,
        #    y compris pour les colonnes créées dans le notebook.
        if c in vocab:
            mods = vocab[c]
            defaut = mods[0]
            if c in cols and cols[c].get("kind") == "categorical":
                d = cols[c].get("default")
                if d in mods:
                    defaut = d
            elif df is not None and c in df.columns and len(df[c].mode()):
                d = str(df[c].mode().iloc[0])
                if d in mods:
                    defaut = d
            # Une colonne à deux modalités 0/1 reste un booléen à l'écran
            if set(mods) <= {"0", "1", "0.0", "1.0", "True", "False"}:
                spec = {"kind": "binary", "default": 0, "source": "modele"}
            else:
                spec = {"kind": "categorical", "values": mods,
                        "default": defaut, "source": "modele"}

        # 2. Colonne numérique du pipeline : bornes issues des données si possible
        elif c in numeriques:
            if df is not None and c in df.columns:
                spec = _infer_spec(df[c], c)
                spec["source"] = "data"
            elif c in cols:
                spec = dict(cols[c])
                spec["source"] = "schema"
            else:
                spec = {"kind": "numeric", "min": 0.0, "max": 1e9,
                        "default": 0.0, "is_int": True, "source": "absent"}

        # 3. Modèle nu (pas de pipeline) : schéma puis données
        elif c in cols:
            spec = dict(cols[c])
            spec["source"] = "schema"
        elif df is not None and c in df.columns:
            spec = _infer_spec(df[c], c)
            spec["source"] = "data"
        else:
            spec = {"kind": "numeric", "min": 0.0, "max": 1e9,
                    "default": 0.0, "is_int": True, "source": "absent"}

        # Un comptage reste entier, quelle que soit la source
        if spec.get("kind") == "numeric" and is_comptage(c):
            spec["is_int"] = True
            spec["default"] = round(float(spec.get("default", 0)))

        # Variable retirée du formulaire (mais conservée pour le modèle)
        spec["hidden"] = c in MASQUEES

        specs[c] = spec

    return specs


def default_for(col: str, spec: dict):
    """Valeur par défaut d'une variable masquée, injectée en coulisses."""
    if spec["kind"] == "categorical":
        return spec.get("default", spec["values"][0] if spec.get("values") else "")
    if spec["kind"] == "binary":
        return int(spec.get("default", 0))
    if spec["kind"] == "text":
        return spec.get("default", "")
    val = spec.get("default", 0)
    return round(float(val)) if spec.get("is_int") else float(val)


def derive(values: dict) -> dict:
    """
    Recalcule les features dérivées quand leurs composantes sont connues.
    Évite de demander à l'utilisateur une valeur qu'on sait calculer.
    """
    v = dict(values)
    if {"est_assure_annulation", "anticipation_jours"} <= set(v):
        v["assure_x_anticip"] = v["est_assure_annulation"] * v["anticipation_jours"]
    return v


# ===========================================================================
# Vocabulaire embarqué dans le pipeline
# ===========================================================================
# Un ColumnTransformer ajusté connaît, pour chaque colonne catégorielle, les
# modalités exactes vues à l'entraînement (`OneHotEncoder.categories_`) ainsi
# que la répartition catégorielles / numériques.
#
# C'est une source plus fiable que `schema.json` : elle vient du modèle
# lui-même, et couvre y compris les colonnes créées dans le notebook et
# absentes du CSV exporté.
# ===========================================================================

def _column_transformers(prep) -> list:
    """Tous les ColumnTransformer présents dans le préprocesseur."""
    if prep is None:
        return []
    if hasattr(prep, "transformers_"):
        return [prep]
    return [s for _, s in getattr(prep, "steps", []) if hasattr(s, "transformers_")]


@st.cache_data(show_spinner=False)
def pipeline_vocab() -> tuple[dict, list]:
    """
    Retourne ({colonne: [modalités exactes]}, [colonnes numériques]).
    Vide si le modèle n'est pas un pipeline avec ColumnTransformer.
    """
    m = load_model()
    if not _is_pipeline(m):
        return {}, []

    vocab: dict = {}
    numeriques: list = []

    for ct in _column_transformers(_preprocessor(m)):
        for _, trans, cols in ct.transformers_:
            if trans == "drop" or cols is None:
                continue

            cols = [cols] if isinstance(cols, str) else list(cols)
            cols = [str(c) for c in cols]

            # Un transformer peut lui-même être un petit pipeline
            enc = trans
            if hasattr(enc, "steps"):
                enc = enc.steps[-1][1]

            categories = getattr(enc, "categories_", None)
            if categories is not None:
                for c, mods in zip(cols, categories):
                    vocab[c] = [str(v) for v in mods]
            else:
                numeriques.extend(cols)

    return vocab, numeriques
