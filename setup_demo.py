"""
Entraîne un modèle de démonstration sur le dataset préparé, pour vérifier
que l'application tourne de bout en bout.

    python prepare_data.py <votre_dataset.csv>    (à lancer AVANT)
    python setup_demo.py [--encodage onehot|ordinal|categorical]

⚠️  Ce modèle est un MODÈLE JOUET. Remplacez `models/xgb_optimise.joblib`
    par votre modèle réel avant la soutenance.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
TARGET = "y_annulation"

p = argparse.ArgumentParser()
p.add_argument("--encodage", default="pipeline",
               choices=["pipeline", "onehot", "ordinal", "categorical"])
args = p.parse_args()

schema_path = ROOT / "data" / "schema.json"
data_path = ROOT / "data" / "dataset_full.parquet"
if not data_path.exists():
    data_path = ROOT / "data" / "sample_dataset.parquet"

if not schema_path.exists() or not data_path.exists():
    raise SystemExit(
        "Lancez d'abord :  python prepare_data.py <votre_dataset.csv>"
    )

schema = json.loads(schema_path.read_text(encoding="utf-8"))
df = pd.read_parquet(data_path)

cols = schema["columns"]
cat = [c for c in cols if cols[c]["kind"] == "categorical"]
feats = list(cols)

X, y = df[feats].copy(), df[TARGET]

if args.encodage == "pipeline":
    # Cas le plus courant en pratique : ColumnTransformer + XGBClassifier.
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num = [c for c in feats if c not in cat]
    taux = y.mean()
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", "passthrough", num),
        ])),
        ("clf", XGBClassifier(
            n_estimators=200, learning_rate=.08, max_depth=4, min_child_weight=8,
            subsample=.8, colsample_bytree=.8, reg_lambda=3.0,
            scale_pos_weight=(1 - taux) / taux,
            eval_metric="aucpr", random_state=42,
        )),
    ])
    pipe.fit(X, y)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "xgb_optimise.joblib")

    print("Structure       : Pipeline scikit-learn (ColumnTransformer + XGBClassifier)")
    print(f"Colonnes entree : {X.shape[1]}")
    print(f"Taux annulation : {taux * 100:.2f} %")
    print("\n✓ models/xgb_optimise.joblib")
    print("\nModèle JOUET — remplacez-le par le vôtre avant la soutenance.")
    raise SystemExit(0)

if args.encodage == "onehot":
    X = pd.get_dummies(X, columns=cat).astype(float)
    # XGBoost refuse les caractères < [ ] dans les noms de features
    X.columns = [re.sub(r"[\[\]<>]", "", c) for c in X.columns]
    kw = {}
elif args.encodage == "categorical":
    for c in cat:
        X[c] = X[c].astype("category")
    kw = {"enable_categorical": True, "tree_method": "hist"}
else:  # ordinal — codes par ordre alphabétique, comme LabelEncoder
    for c in cat:
        m = {v: i for i, v in enumerate(cols[c]["values"])}
        X[c] = X[c].astype(str).map(m)
    kw = {}

taux = y.mean()
model = XGBClassifier(
    n_estimators=200, learning_rate=.08, max_depth=4, min_child_weight=8,
    subsample=.8, colsample_bytree=.8, reg_lambda=3.0,
    scale_pos_weight=(1 - taux) / taux,
    eval_metric="aucpr", random_state=42, **kw,
)
model.fit(X, y)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "xgb_optimise.joblib")

print(f"Encodage        : {args.encodage}")
print(f"Features        : {X.shape[1]}")
print(f"Taux annulation : {taux * 100:.2f} %")
print("\n✓ models/xgb_optimise.joblib")
print("\nModèle JOUET — remplacez-le par le vôtre avant la soutenance.")
