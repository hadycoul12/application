# Exporter le jeu de données d'entraînement

L'application a besoin de connaître **toutes** les colonnes que votre modèle réclame — y compris celles que vous avez créées dans le notebook (`est_solo`, `a_prestations`, `anticipation_tranche`, `assure_x_anticip`…).

Si vous générez `schema.json` à partir de `dataset_annulation_clean.csv`, ces colonnes manquent : elles n'existent que dans votre notebook, après feature engineering. L'application les signale alors comme *introuvables* et le score est dégradé.

## Le correctif, en une cellule

Dans `modele_v2.ipynb`, **juste avant le `model.fit()`**, ajoutez cette cellule :

```python
#  Export pour l'application Streamlit
# X est le DataFrame passé au fit, y la cible.
# On exporte exactement ce que le modèle voit, features dérivées comprises.

export = X.copy()
export["y_annulation"] = y

export.to_csv("dataset_entrainement.csv", index=False, encoding="utf-8")
print(f"{len(export):,} lignes · {len(export.columns)} colonnes exportées")
print(f"Colonnes : {', '.join(export.columns)}")
```

Adaptez `X` et `y` aux noms réellement utilisés dans votre notebook (souvent `X_train`/`y_train`, ou `df[FEATURES]`/`df[TARGET]`).

## Puis, dans l'application

```bash
python prepare_data.py "C:/chemin/vers/dataset_entrainement.csv"
```

Toutes les colonnes du formulaire passeront alors en source **« schema »**, et le diagnostic d'accueil n'affichera plus aucune colonne introuvable.

## Vérification

Ouvrez **Accueil → Diagnostic technique du modèle**. Le tableau des colonnes doit afficher, pour chacune, une origine parmi :

| Origine | Signification |
|---|---|
| Modalités lues dans le modèle | Idéal — modalités exactes de l'encodeur ajusté |
| Décrite dans schema.json | Correct |
| Retrouvée dans les données | Correct |
| **Introuvable** | **À corriger** — refaites l'export ci-dessus |

Tant qu'il reste des colonnes *introuvables*, le score affiché ne reflète pas ce que produirait votre modèle en conditions réelles.
