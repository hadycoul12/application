"""Interprétabilité SHAP des prédictions."""

import re

import shap
import streamlit as st

from utils.model import encoding_plan, label, transformed_frame


@st.cache_resource(show_spinner=False)
def _explainer(_est):
    """TreeExplainer sur l'estimateur terminal (jamais sur le Pipeline)."""
    return shap.TreeExplainer(_est)


def _pretty(feature: str) -> str:
    """
    Rend lisible un nom de feature encodée.

    « canal_CE Global »              → « Canal de vente : CE Global »
    « cat__canal_CE Global »         → idem (préfixe ColumnTransformer retiré)
    « num__anticipation_jours »      → « Anticipation »
    """
    f = re.sub(r"^[A-Za-z0-9]+__", "", feature)   # préfixe ColumnTransformer

    if f in ("", None):
        return feature

    plan = encoding_plan()
    for col in plan["cat_cols"]:
        if f.startswith(f"{col}_") and len(f) > len(col) + 1:
            return f"{label(col)} : {f[len(col) + 1:]}"

    return label(f)


def shap_drivers(values: dict, top: int = 8) -> list[dict]:
    """
    Principaux facteurs de la prédiction, triés par impact absolu.

    [{"feature", "label", "value", "shap", "direction"}, ...]
    """
    X, est, noms = transformed_frame(values)
    sv = _explainer(est).shap_values(X)[0]

    drivers = [
        {
            "feature": nom,
            "label": _pretty(nom),
            "value": X.iloc[0][col],
            "shap": float(v),
            "direction": "up" if v > 0 else "down",
        }
        for col, nom, v in zip(X.columns, noms, sv)
    ]

    # Sur des features one-hot, une modalité inactive (valeur 0) n'apporte
    # rien d'interprétable au gestionnaire : on ne garde que les modalités
    # effectivement portées par le dossier.
    actifs = [
        d for d in drivers
        if not (" : " in d["label"] and float(d["value"]) == 0)
    ]
    if actifs:
        drivers = actifs

    drivers.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return drivers[:top]
