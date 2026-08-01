# Projet — Scoring d'annulation (Mémoire M2, Maeva / Pierre & Vacances)

Application Streamlit multi-pages de scoring prédictif du risque d'annulation de
réservation. Développée dans le cadre d'un mémoire de Master 2 Data & IA (Nexa
Digital School), en alternance chez Maeva.

**Stack** : Streamlit (`st.navigation`), XGBoost ou Pipeline scikit-learn, SHAP,
Plotly, pandas/pyarrow (Parquet).

**Lancer l'app** : `streamlit run app.py` — login `maeva` / `maeva2026` (voir
`.streamlit/secrets.toml`, à créer depuis `secrets.example.toml`).

---

## Principe fondateur : tout est piloté par les données et le modèle

**Aucune modalité, aucun libellé, aucun encodage n'est codé en dur dans
l'application.** C'est la règle la plus importante du projet — elle a été
introduite après plusieurs bugs (`ValueError: int('Avt Hiver')`, canaux inventés
ne correspondant pas aux vraies données, colonnes manquantes au scoring).

Trois sources de vérité, dans cet ordre de priorité :

1. **Le modèle lui-même** (`utils/model.py::pipeline_vocab`, `encoding_plan`) —
   pour un Pipeline scikit-learn, les modalités exactes vues à l'entraînement
   sont lues dans `ColumnTransformer.transformers_` / `OneHotEncoder.categories_`.
   C'est la référence absolue : elle couvre même les colonnes créées dans le
   notebook (feature engineering) et absentes du CSV exporté.
2. **`data/schema.json`** — généré par `prepare_data.py` à partir du dataset
   réel. Décrit chaque colonne : type (`categorical` / `binary` / `numeric`),
   modalités réelles, bornes, valeur par défaut.
3. **Le dataset chargé** (`data/dataset_full.parquet` ou `sample_dataset.parquet`)
   — utilisé en dernier recours pour inférer une spec quand une colonne est
   absente du schéma mais présente dans les données.

`utils/model.py::field_specs()` fusionne ces trois sources dans cet ordre et
alimente **tout le reste** : formulaire de prédiction, filtres du dashboard,
colonnes attendues en scoring batch. Ne jamais construire un formulaire ou un
filtre à partir d'une liste de modalités écrite en dur dans le code.

---

## Architecture

```
app.py                      Point d'entrée : consentement RGPD → login → navigation
prepare_data.py             CSV brut → schema.json + dataset_full.parquet + sample_dataset.parquet
setup_demo.py                Entraîne un modèle JOUET pour tester l'app (3 modes d'encodage)
EXPORT_NOTEBOOK.md           Guide pour exporter le dataframe d'entraînement depuis le notebook

utils/
  auth.py                    Bannière consentement RGPD (bloquante) + login + journal d'audit
  model.py                   ★ Cœur du projet — voir section dédiée ci-dessous
  explain.py                 SHAP (TreeExplainer), libellés lisibles des features encodées
  ui.py                      render_html(), page_header(), sections, logo, sidebar

views/
  accueil.py                 KPIs modèle + diagnostic technique (encodage détecté, colonnes)
  dashboard.py                EDA en 5 onglets, piloté par schema.json (aucune modalité en dur)
  prediction.py               Formulaire piloté par field_specs() + scoring + SHAP
  batch.py                    Upload CSV → scoring de masse → export
  impact.py                   Simulation CA sauvé selon le seuil de décision
  historique.py                Journal des prédictions effectuées
  rgpd.py                      Minimisation, sécurité, registre des traitements, audit log

assets/style.css              Design system (sidebar navy, hero en dégradé, Fraunces + Inter)
data/schema.json              Généré — NE PAS éditer à la main, relancer prepare_data.py
models/xgb_optimise.joblib    Le modèle (XGBoost nu OU Pipeline sklearn — les deux sont gérés)
```

---

## `utils/model.py` — à comprendre avant d'y toucher

C'est le fichier le plus sensible du projet. Il gère **trois structures de
modèle possibles**, détectées automatiquement (jamais supposées) :

| Structure détectée | Comment on la reconnaît | Comment on l'utilise |
|---|---|---|
| **Pipeline scikit-learn** | `hasattr(model, "steps")` ou similaire | Colonnes brutes passées telles quelles ; le préprocessing (OneHotEncoder, etc.) est embarqué dans le pipeline. **Cas le plus fiable.** |
| **XGBoost + one-hot** | `feature_names` contient des colonnes du type `canal_Booking` absentes du schéma | `pd.get_dummies()` puis réindexation sur les colonnes attendues |
| **XGBoost + encodage ordinal** | Ni pipeline ni one-hot détecté | Codes reconstruits par ordre alphabétique des modalités (= comportement de `LabelEncoder`/`.cat.codes`) |
| **XGBoost + catégoriel natif** | `feature_types` contient `"c"` | Colonnes passées en `dtype="category"` |

Fonctions clés :

- `encoding_plan()` — détecte la structure, retourne `mode`, `expected`
  (colonnes/features attendues), `base_cols` (colonnes sources à fournir).
- `pipeline_vocab()` — pour un Pipeline, extrait `{colonne: [modalités exactes]}`
  directement de l'encodeur ajusté. **Source de vérité n°1.**
- `field_specs()` — construit la spec de saisie de CHAQUE colonne réclamée par
  le modèle, en fusionnant `pipeline_vocab()` → `schema.json` → données brutes.
  Chaque entrée porte un champ `source` (`modele` / `schema` / `data` / `absent`)
  utilisé pour tracer d'où vient l'information (affiché dans le diagnostic
  d'Accueil).
- `align(df)` — transforme un DataFrame brut en la matrice exacte attendue par
  le modèle, quel que soit le mode détecté.
- `MASQUEES` — ensemble de colonnes retirées du FORMULAIRE de prédiction (ex :
  assurance annulation, groupe fournisseur, type de produit, client VIP) mais
  **toujours utilisées par le modèle** : elles reçoivent une valeur par défaut
  injectée en coulisses (`default_for()`). Pour en ajouter/retirer, éditer cet
  ensemble.
- `COMPTAGES` / `is_comptage()` — force certaines colonnes (nb_bebe, nb_mineur,
  nb_dossiers_anterieurs…) à rester des entiers dans les formulaires, même si le
  CSV les stocke en float.

**Si une colonne attendue par le modèle est absente du dataset ET du schéma**,
elle apparaît avec `source: "absent"` : l'app affiche un avertissement explicite
plutôt que de produire un score silencieusement faux. C'est volontaire — ne pas
« corriger » en masquant l'avertissement.

---

## Règles impératives

### 1. Aucune modalité codée en dur
Ne jamais écrire `{"canal": {0: "Site Maeva", 1: "Booking", ...}}` ou équivalent
dans le code. Toute liste de modalités doit provenir de `field_specs()`,
`pipeline_vocab()` ou `schema.json`. Si une page a besoin de connaître les
canaux existants, elle les lit dynamiquement — jamais une constante.

### 2. Le HTML passe systématiquement par `render_html()`
Ne jamais utiliser `st.markdown("""...multi-ligne indenté...""", unsafe_allow_html=True)`.
**Piège découvert et corrigé** : Streamlit interprète tout bloc indenté de 4+
espaces comme un bloc de code, ce qui affiche le HTML en texte brut au lieu de
le rendre. `utils/ui.py::render_html()` aplatit le HTML sur une ligne avant
rendu — toujours passer par cette fonction pour du HTML custom (bannières,
cartes, badges).

### 3. Pas de chunking ni de downcast agressif sur les CSV
**Bug corrigé** : un ancien `prepare_data.py` lisait le CSV par morceaux de
50 000 lignes avec `pd.to_numeric(..., downcast="integer")`. Le premier morceau
ne contenant que des valeurs ≤ 127 forçait un typage `int8`, qui débordait dès
qu'une valeur ultérieure dépassait 127 (`ArrowInvalid: Integer value 155 not in
range`). Le fichier actuel lit le CSV en une seule passe, sans downcast — ~310k
lignes tiennent largement en mémoire (quelques dizaines de Mo). Ne pas
réintroduire de chunking sans une vraie raison de volumétrie.

### 4. Détection de type : `is_numeric_dtype`, jamais `dtype == object`
**Bug corrigé** : pandas récent (2.x/3.x) expose le dtype `str` pour les
colonnes texte, plus `object`. Un test `s.dtype == object` classait alors les
colonnes catégorielles comme numériques et plantait sur `np.nanmin()`. La
détection correcte : `not pd.api.types.is_numeric_dtype(s)` → catégoriel.

### 5. Toujours tester avec `streamlit.testing.v1.AppTest` avant de valider
Chaque page peut être testée hors navigateur :

```python
from streamlit.testing.v1 import AppTest
t = AppTest.from_file("views/prediction.py", default_timeout=300)
t.session_state["consent"] = True
t.session_state["auth"] = True
t.session_state["user"] = "maeva"
t.run()
assert not t.exception
```

Pour simuler un scoring : `t.selectbox(key="f_canal").set_value(...)`, puis
`t.button[0].click().run()`. Vérifier `t.exception` et `t.error`. C'est le
moyen le plus rapide de détecter une régression sans lancer un vrai serveur.

### 6. Le modèle dicte le formulaire, jamais l'inverse
`views/prediction.py` et `views/batch.py` construisent leurs champs à partir de
`field_specs()` — c'est-à-dire des colonnes que **le modèle réclame** — et non
à partir de `schema.json` seul. Une colonne peut exister dans le schéma sans
être utilisée par le modèle (et inversement, pour les features dérivées créées
dans le notebook).

---

## Variables dérivées et masquées

- `assure_x_anticip` = `est_assure_annulation × anticipation_jours` — recalculée
  automatiquement (`utils/model.py::derive()`), jamais saisie par l'utilisateur.
- Variables dans `MASQUEES` (assurance annulation, groupe fournisseur, type de
  produit, client VIP) : absentes du formulaire de prédiction, valeur par défaut
  injectée via `default_for()`. Le modèle continue de les utiliser pour scorer.

Si le modèle réclame une colonne dérivée dans le notebook (ex :
`anticipation_tranche`, `est_solo`) et qu'elle n'existe pas dans le CSV brut,
voir `EXPORT_NOTEBOOK.md` : la solution propre est d'exporter le DataFrame
**juste avant `model.fit()`**, features dérivées comprises, et de relancer
`prepare_data.py` dessus.

---

## Design system

- Police d'affichage : **Fraunces** (serif éditorial) pour les titres, **Inter**
  pour le corps. Palette : encre navy (`--ink-950` à `--ink-700`) pour la
  sidebar et les bannières hero, corail (`--brand: #E8593C`) comme accent Maeva.
- Bannière hero en dégradé navy → teal, avec pastille de section en majuscules.
- Logo Maeva : `utils/ui.py::find_logo()` cherche `assets/logo_maeva.{png,jpg,svg}`
  (ou à la racine du projet) et l'injecte en data-URI. Sans logo, repli textuel
  automatique — ne jamais faire planter l'app si le fichier est absent.
- Toutes les couleurs/styles sont dans `assets/style.css`, chargé une fois par
  page via `inject_css()` (appelé dans `utils/auth.py::guard()`).

---

## Ne pas toucher / ne pas committer

- `data/dataset_full.parquet` — **données commerciales réelles Maeva**,
  confidentielles. Exclu par `.gitignore`. Ne jamais le pousser sur un repo
  public, même privé sans vérification préalable.
- `.streamlit/secrets.toml` — identifiants de connexion, exclu par `.gitignore`.
  `secrets.example.toml` est le seul fichier à versionner.
- `data/audit_log.csv`, `data/predictions_history.csv` — générés à l'exécution,
  exclus par `.gitignore`.
- `models/xgb_optimise.joblib` — le vrai modèle entraîné sur données réelles.
  À versionner avec prudence (poids du fichier, confidentialité du modèle
  lui-même selon la politique Maeva — à clarifier si besoin).

## Peut être régénéré sans risque

- `data/schema.json`, `data/sample_dataset.parquet` → `python prepare_data.py <csv>`
- `models/xgb_optimise.joblib` (version jouet) → `python setup_demo.py [--encodage onehot|ordinal|categorical]`

---

## Prochaines pistes (non commencées)

- Étendre le formulaire de prédiction pour couvrir plus de colonnes d'engagement
  email si le dataset CRM Batch est disponible.
- Vérifier si `models/xgb_optimise.joblib` doit être versionné ou distribué
  séparément (poids, confidentialité).
- Éventuellement migrer `data/predictions_history.csv` vers une base persistante
  (SQLite) si le déploiement Streamlit Cloud doit conserver l'historique entre
  redéploiements.
