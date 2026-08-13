"""
Prépare les données de l'application à partir du dataset complet.

    python prepare_data.py "C:/chemin/vers/dataset_annulation_clean.csv"

Produit :

  data/schema.json              type et modalités RÉELLES de chaque colonne
  data/dataset_full.parquet     jeu complet, LOCAL uniquement, ignoré par Git
  data/sample_dataset.parquet   échantillon stratifié, versionné, sert au déploiement

Le schéma est la pièce maîtresse : aucune modalité n'est codée en dur dans
l'application. Les libellés affichés (« CE Global », « Avt Hiver ») proviennent
directement de vos données.

Note : le fichier est lu en une seule passe. Un CSV de 310 000 lignes pèse une
vingtaine de mégaoctets, le découper en morceaux n'apporterait rien et
introduirait des incohérences de typage entre morceaux.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

TARGET = "y_annulation"
TAILLE_ECHANTILLON = 30_000
MAX_MODALITES = 60

# Colonnes qui ne doivent jamais subsister
INTERDITES = {
    "client_email", "email", "nom", "prenom", "telephone", "tel",
    "adresse", "adresse_postale", "code_postal_client", "date_naissance",
}

# Colonnes techniques : conservées dans les données, exclues des features
TECHNIQUES = {"dossier_cle", "id_dossier", "index", "unnamed: 0"}


def kind_of(s: pd.Series) -> str:
    """Type d'une colonne : categorical, binary ou numeric."""
    # pandas récent expose le dtype `str` et non plus `object` : on teste donc
    # la numéricité, et tout le reste est considéré comme catégoriel.
    if not pd.api.types.is_numeric_dtype(s):
        return "categorical"
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    vals = set(pd.unique(s.dropna()))
    if vals and vals <= {0, 1, 0.0, 1.0, True, False}:
        return "binary"
    return "numeric"


def main(src: str):
    path = Path(src)
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")

    # Lecture
    print(f"Lecture de {path.name} …")
    apercu = path.read_text(encoding="utf-8-sig", errors="ignore")[:4000]
    sep = ";" if apercu.count(";") > apercu.count(",") else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,} lignes · {len(df.columns)} colonnes".replace(",", " "))

    # Contrôle RGPD
    a_jeter = [c for c in df.columns if c.lower() in INTERDITES]
    if a_jeter:
        df = df.drop(columns=a_jeter)
        print(f"   Colonnes identifiantes supprimées : {', '.join(a_jeter)}")
    else:
        print("  ✓ Aucune colonne directement identifiante détectée.")

    if TARGET not in df.columns:
        sys.exit(
            f"\n✗ La colonne cible `{TARGET}` est absente.\n"
            f"  Colonnes disponibles : {', '.join(df.columns)}"
        )

    taux = df[TARGET].mean()
    print(f"  Taux d'annulation : {taux * 100:.2f} %\n")

    # Découverte du schéma
    schema = {"target": TARGET, "n_rows": int(len(df)), "columns": {}}
    ignorees = []

    for col in df.columns:
        if col == TARGET or col.lower() in TECHNIQUES:
            continue

        s = df[col]
        k = kind_of(s)

        if k == "categorical":
            mods = sorted(str(v) for v in s.dropna().unique())
            if len(mods) > MAX_MODALITES:
                ignorees.append(f"{col} ({len(mods)} modalités)")
                continue
            # L'ordre alphabétique est celui qu'appliquent LabelEncoder et
            # .cat.codes : il permet de reconstituer l'encodage numérique.
            schema["columns"][col] = {
                "kind": "categorical",
                "values": mods,
                "default": str(s.mode().iloc[0]) if len(s.mode()) else mods[0],
            }

        elif k == "binary":
            schema["columns"][col] = {
                "kind": "binary",
                "default": int(s.mode().iloc[0]) if len(s.mode()) else 0,
            }

        else:
            schema["columns"][col] = {
                "kind": "numeric",
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                "default": float(np.nanmedian(s)),
                "is_int": bool(pd.api.types.is_integer_dtype(s)),
            }

    n_cat = sum(1 for v in schema["columns"].values() if v["kind"] == "categorical")
    n_bin = sum(1 for v in schema["columns"].values() if v["kind"] == "binary")
    n_num = sum(1 for v in schema["columns"].values() if v["kind"] == "numeric")
    print(f"Schéma découvert : {n_cat} catégorielles · {n_bin} binaires · {n_num} numériques")

    for col, spec in schema["columns"].items():
        if spec["kind"] == "categorical":
            apercu_mods = ", ".join(spec["values"][:4])
            suite = f"… (+{len(spec['values']) - 4})" if len(spec["values"]) > 4 else ""
            print(f"  · {col:<26} {apercu_mods}{suite}")

    if ignorees:
        print(f"\n  Colonnes ignorées (trop de modalités) : {', '.join(ignorees)}")

    (DATA / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n✓ schema.json")

    # Écriture Parquet 

    full = DATA / "dataset_full.parquet"
    df.to_parquet(full, index=False, compression="snappy")
    print(f"✓ dataset_full.parquet    ({len(df):,} lignes · "
          f"{full.stat().st_size / 1e6:.1f} Mo)".replace(",", " "))

    if len(df) > TAILLE_ECHANTILLON:
        frac = TAILLE_ECHANTILLON / len(df)
        ech = (
            pd.concat([g.sample(frac=frac, random_state=42) for _, g in df.groupby(TARGET)])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    else:
        ech = df.copy()

    samp = DATA / "sample_dataset.parquet"
    ech.to_parquet(samp, index=False, compression="snappy")
    print(f"✓ sample_dataset.parquet  ({len(ech):,} lignes · "
          f"{samp.stat().st_size / 1e6:.1f} Mo · taux {ech[TARGET].mean() * 100:.2f} %)"
          .replace(",", " "))

    print(
        "\nÀ retenir :\n"
        "  · dataset_full.parquet est ignoré par Git, il reste sur votre machine.\n"
        "    L'application le charge automatiquement (démo de soutenance).\n"
        "  · sample_dataset.parquet et schema.json sont versionnés et alimentent\n"
        "    l'application déployée."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python prepare_data.py <chemin/vers/dataset.csv>")
    main(sys.argv[1])
