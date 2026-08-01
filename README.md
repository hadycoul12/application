# Scoring prédictif du risque d'annulation

Application web de scoring des annulations de réservation, développée dans le cadre du mémoire de Master 2 Data & Intelligence Artificielle (Nexa Digital School) — alternance chez Maeva, groupe Pierre & Vacances.

Un modèle XGBoost attribue à chaque réservation une probabilité d'annulation, explique sa décision via les valeurs SHAP, et permet d'arbitrer le seuil d'alerte au regard de son impact business.

---

## Installation

```bash
git clone https://github.com/hadycoul12/scoring-annulation-maeva.git
cd scoring-annulation-maeva

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Configuration des identifiants

Streamlit lit **uniquement** `.streamlit/secrets.toml`. Le fichier `.example` n'est qu'un modèle : l'éditer ne suffit pas.

```bash
copy .streamlit\secrets.example.toml .streamlit\secrets.toml    # Windows
# cp .streamlit/secrets.example.toml .streamlit/secrets.toml    # macOS / Linux
```

```toml
[credentials]
username = "maeva"
password = "maeva2026"
```

### Données — à faire en premier

```bash
python prepare_data.py "C:/chemin/vers/dataset_annulation_clean.csv"
```

Le fichier est lu **par morceaux** et jamais chargé entièrement en mémoire : le script fonctionne donc sur une machine modeste, ou avec un interpréteur Python 32 bits (plafonné à environ 2 Go). Pic mesuré : 256 Mo sur un CSV de 310 000 lignes.

Si la mémoire manque malgré tout, réduisez la taille des morceaux :

```bash
python prepare_data.py "…/dataset_annulation_clean.csv" --chunk 10000
```

Le script **découvre le schéma dans vos données** et produit trois fichiers :

| Fichier | Contenu | Git |
|---|---|---|
| `data/schema.json` | Type et modalités réelles de chaque colonne | Versionné |
| `data/dataset_full.parquet` | Jeu complet (~310 k lignes, ~4 Mo) | **Ignoré** — reste sur votre machine |
| `data/sample_dataset.parquet` | Échantillon stratifié (30 k lignes) | Versionné |

`schema.json` est la pièce maîtresse : **aucune modalité n'est codée en dur dans l'application**. Les libellés affichés dans les formulaires et les graphiques (« CE Global », « Avt Hiver », « NoFlex <J30 ») proviennent directement de votre dataset. Si vos modalités changent, il suffit de relancer ce script.

Le script contrôle également qu'aucune colonne identifiante ne subsiste (`client_email`, `nom`, `telephone`…) et les supprime le cas échéant.

> **Confidentialité.** Ne poussez jamais `dataset_full.parquet` sur un dépôt public : il contient des données commerciales réelles couvertes par l'accord Maeva. Le `.gitignore` l'exclut par défaut.

### Modèle

Déposez votre modèle entraîné dans `models/xgb_optimise.joblib`.

**La structure du modèle est détectée automatiquement.** Quatre cas sont pris en charge :

| Structure | Détection | Traitement |
|---|---|---|
| `Pipeline` scikit-learn | L'objet possède `.steps` | Les colonnes brutes sont transmises — le pipeline encode lui-même |
| One-hot (`get_dummies`) | Le modèle attend `canal_Booking`, `canal_Sites web`… | `get_dummies` puis réalignement |
| Numérique (`LabelEncoder`, `.cat.codes`) | Codes par ordre alphabétique des modalités | Mapping reconstitué depuis `schema.json` |
| Catégoriel natif XGBoost | `feature_types` contient `c` | Conversion en `category` |

Le cas `Pipeline` est le plus fiable : le préprocessing étant embarqué dans le `.joblib`, aucune reconstitution d'encodage n'est nécessaire. Pour les explications SHAP, l'application applique le préprocessing puis interroge l'estimateur terminal, `TreeExplainer` ne sachant pas travailler sur un pipeline.

Le mode détecté est affiché dans **Accueil → Diagnostic technique du modèle**. Vérifiez-le avant la soutenance. En cas d'incompatibilité, la prédiction échoue explicitement plutôt que de produire un score faux.

Pour tester l'interface sans modèle réel :

```bash
python setup_demo.py                      # encodage numérique
python setup_demo.py --encodage onehot    # ou one-hot
```

> Ce script produit un modèle **jouet**. Remplacez-le par le vôtre avant la soutenance.

### Lancement

```bash
streamlit run app.py
```

---

## Adapter le schéma des features

L'ordre des colonnes passées au modèle doit correspondre **exactement** à celui de l'entraînement. Pour le vérifier :

```python
import joblib
model = joblib.load("models/xgb_optimise.joblib")
print(model.get_booster().feature_names)
```

Le dictionnaire `SCHEMA` dans `utils/model.py` est la source de vérité unique : il définit l'ordre des features, les libellés affichés, les widgets de saisie et leur regroupement en onglets. Modifier ce dictionnaire suffit à répercuter le changement sur l'ensemble de l'application.

---

## Architecture

```
.
├── app.py                    Point d'entrée — navigation, garde RGPD + auth
├── prepare_data.py           Conversion du dataset complet en Parquet + échantillon
├── views/
│   ├── accueil.py            Présentation et performances du modèle
│   ├── dashboard.py          Exploration du portefeuille + EDA (Plotly)
│   ├── prediction.py         Scoring unitaire + explication SHAP
│   ├── batch.py              Scoring par lot (import / export CSV)
│   ├── impact.py             Simulation du CA sauvé, arbitrage du seuil
│   ├── historique.py         Traçabilité des prédictions
│   └── rgpd.py               Minimisation, sécurité, registre, journal d'audit
├── utils/
│   ├── auth.py               Consentement, authentification, journal d'audit
│   ├── model.py              Schéma des features, chargement, prédiction
│   ├── explain.py            Valeurs SHAP
│   └── ui.py                 Composants d'interface
├── assets/style.css          Design system
├── models/                   Modèle sérialisé (.joblib)
└── data/                     Échantillon anonymisé, journaux
```

Les noms de fichiers sont volontairement en ASCII : la navigation est déclarée dans `app.py` via `st.navigation`, et les icônes y sont définies dans le code. Les émojis dans les noms de fichiers provoquent des erreurs d'encodage sous Windows.

---

## Performances du modèle

| Métrique | Valeur | Lecture |
|---|---|---|
| PR-AUC | 0.263 | ×3.6 par rapport à la baseline aléatoire (7.27 %) |
| Rappel | 62.7 % | 2 835 annulations captées sur 4 522 |
| Précision | 16.8 % | ×2.3 par rapport à la prévalence de base |
| AUC-ROC | 0.764 | — |
| Seuil retenu | 0.50 | Maximise le F1 sous contrainte de rappel ≥ 60 % |

XGBoost optimisé par RandomizedSearchCV (100 itérations, validation croisée 5-fold), pondération native des classes sans rééchantillonnage synthétique. Évaluation sur un holdout stratifié de 20 %.

---

## Tableau de bord

L'analyse exploratoire est répartie en cinq onglets :

| Onglet | Contenu |
|---|---|
| Vue d'ensemble | Déséquilibre de la cible, taux par canal, cartographie volume × risque |
| Facteurs structurels | Effet de la condition Flex, variables binaires, interaction assurance × anticipation |
| Temporalité | Gradient d'anticipation, saisonnalité, durée de séjour, distribution des délais |
| Engagement email | Couverture CRM, effet des campagnes et de la récence sur le sous-ensemble couvert |
| Corrélations | Corrélation de chaque variable avec la cible, matrice de corrélation |

---

## Conformité RGPD

| Mesure | Implémentation |
|---|---|
| Consentement | Bannière bloquante à l'entrée — aucun traitement avant acceptation |
| Minimisation | 7 variables identifiantes exclues dès l'export BigQuery |
| Sécurité | HTTPS, authentification obligatoire, journal d'audit horodaté |

Les variables `client_email`, `nom`, `prenom`, `telephone`, `adresse_postale`, `code_postal_client` et `date_naissance` sont écartées du dataset et du modèle. Seul l'identifiant technique `dossier_cle` est conservé.

---


## Personnalisation

### Logo Maeva

Déposez `logo_maeva.png` (ou `.jpg`, `.svg`) dans le dossier `assets/` — ou à la
racine du projet. Il apparaît automatiquement sur la bannière d'accueil et dans
la barre latérale. Sans logo, un intitulé texte s'affiche à la place.

### Variables retirées du formulaire de prédiction

Certaines variables restent utilisées par le modèle mais ne sont pas demandées
au gestionnaire (valeur peu actionnable ou déjà connue du système) : assurance
annulation, groupe fournisseur, type de produit, client VIP. Elles reçoivent une
valeur par défaut en coulisses. Pour modifier cette liste, éditez l'ensemble
`MASQUEES` dans `utils/model.py`.

### Comptages

Les colonnes de comptage (nombre de bébés, de mineurs, de voyageurs…) sont
forcées en entier dans les formulaires, même si le CSV les stocke en décimal.
La règle est centralisée dans `is_comptage()` / `COMPTAGES` (`utils/model.py`).

## Déploiement sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub — vérifier que `.streamlit/secrets.toml` est bien ignoré par Git.
2. Sur [share.streamlit.io](https://share.streamlit.io), créer une application pointant vers ce dépôt, fichier principal `app.py`.
3. Dans *Settings → Secrets*, coller le bloc `[credentials]`.
4. Déployer.

---

## Dépannage

**« Identifiant ou mot de passe incorrect » alors que les identifiants sont bons.**
Le fichier édité est probablement `secrets.example.toml` et non `secrets.toml`. Vérifier également que la commande `streamlit run app.py` est lancée depuis la racine du projet — Streamlit cherche `.streamlit/secrets.toml` relativement au répertoire courant.

**Le menu latéral affiche des caractères illisibles.**
Symptôme d'émojis dans les noms de fichiers. Cette version n'en contient aucun : supprimer l'ancien dossier `pages/` s'il subsiste.

**« Colonnes manquantes » lors d'un scoring par lot.**
Le CSV ne contient pas toutes les features attendues. La liste exacte figure dans la section « Format de fichier attendu » de la page *Scoring batch*. La colonne `assure_x_anticip` est recalculée automatiquement si `est_assure_annulation` et `anticipation_jours` sont présentes.

---

## Auteur

**Hady Coulibaly** — Master 2 Data & Intelligence Artificielle
[LinkedIn](https://linkedin.com/in/hady-coulibaly) · [GitHub](https://github.com/hadycoul12)

Projet académique. Les données réelles restent soumises à l'accord de confidentialité Maeva.
