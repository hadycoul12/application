# Scoring prédictif du risque d'annulation

Application web qui estime, pour chaque réservation, la probabilité qu'elle soit annulée. À partir de ce score, l'outil aide un gestionnaire à repérer les dossiers à risque et à décider où concentrer ses actions de rétention.

Le projet a été réalisé dans le cadre de mon mémoire de Master 2 Data et Intelligence Artificielle (Nexa Digital School), en alternance chez Maeva (groupe Pierre et Vacances Center Parcs).

Sous le capot, un modèle XGBoost attribue une probabilité calibrée à chaque dossier, explique sa décision avec les valeurs SHAP, et propose une lecture business du seuil d'alerte.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé :

- **Python 3.10 ou une version plus récente** (testé avec 3.10 et 3.12)
- **pip**, le gestionnaire de paquets Python (livré avec Python)
- **Git**, pour récupérer le dépôt
- Environ **500 Mo d'espace disque** pour l'environnement virtuel et les dépendances

Pour vérifier votre version de Python :

```bash
python --version
```

## Installation

### 1. Récupérer le projet

```bash
git clone https://github.com/hadycoul12/application.git
cd application
```

### 2. Créer et activer un environnement virtuel

L'environnement virtuel isole les dépendances du projet du reste de votre machine.

```bash
python -m venv .venv
```

Puis activez-le selon votre système :

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# macOS ou Linux
source .venv/bin/activate
```

Une fois activé, le nom `(.venv)` apparaît au début de la ligne de commande.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

L'installation télécharge Streamlit, XGBoost, scikit-learn, SHAP, Plotly et quelques autres bibliothèques. Comptez deux à quatre minutes selon votre connexion.

### 4. Configurer les identifiants

L'accès à l'application est protégé par un mot de passe. Streamlit lit ces informations **uniquement** dans le fichier `.streamlit/secrets.toml`. Le fichier `secrets.example.toml` fourni n'est qu'un modèle : le modifier ne suffit pas, il faut en créer une copie.

```bash
# Windows
copy .streamlit\secrets.example.toml .streamlit\secrets.toml

# macOS ou Linux
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Le fichier créé contient déjà les identifiants de test :

```toml
[credentials]
username = "maeva"
password = "maeva2026"
```

## Identifiants de test

Une fois l'application lancée, connectez-vous avec :

| Champ | Valeur |
|---|---|
| Identifiant | `maeva` |
| Mot de passe | `maeva2026` |

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`. Un écran de consentement RGPD s'affiche d'abord, puis la page de connexion.

Le dépôt contient déjà un échantillon de données et le modèle entraîné : l'application fonctionne donc immédiatement, sans étape supplémentaire.

## Données et modèle

Trois fichiers permettent à l'application de tourner, tous présents dans le dépôt :

| Fichier | Rôle |
|---|---|
| `data/sample_dataset.parquet` | Échantillon stratifié servant à l'exploration et à la démonstration |
| `data/schema.json` | Description des colonnes (types et modalités réelles), pour ne rien coder en dur |
| `models/xgb_optimise.joblib` | Le modèle entraîné, accompagné de son calibrateur |

Pour régénérer le schéma et l'échantillon à partir de votre propre jeu de données :

```bash
python prepare_data.py "chemin/vers/votre_dataset.csv"
```

Le jeu complet (`data/dataset_full.parquet`) reste sur votre machine et n'est jamais versionné : il contient des données commerciales réelles couvertes par l'accord de confidentialité Maeva.

## Structure du projet

```
application/
├── app.py                Point d'entrée : consentement RGPD, authentification, navigation
├── prepare_data.py       Génération du schéma et de l'échantillon à partir d'un CSV
├── requirements.txt      Dépendances Python
├── views/
│   ├── accueil.py        Présentation de l'outil et performances du modèle
│   ├── dashboard.py      Exploration du portefeuille et analyse exploratoire
│   ├── prediction.py     Scoring d'un dossier et explication SHAP
│   ├── batch.py          Scoring d'un lot par import de CSV
│   ├── impact.py         Simulation du chiffre d'affaires préservé
│   ├── historique.py     Journal des prédictions
│   └── rgpd.py           Minimisation, sécurité, registre, journal d'audit
├── utils/
│   ├── auth.py           Consentement, authentification, audit
│   ├── model.py          Chargement du modèle, schéma des features, prédiction
│   ├── explain.py        Calcul des valeurs SHAP
│   └── ui.py             Composants d'interface
├── assets/style.css      Feuille de style
├── models/               Modèle et calibrateur (.joblib)
└── data/                 Échantillon, schéma, journaux générés à l'exécution
```

## Fonctionnalités

- **Accueil** : vue d'ensemble des performances et diagnostic technique du modèle.
- **Dashboard** : analyse exploratoire du portefeuille (condition tarifaire, canal, saisonnalité, engagement email, corrélations).
- **Prédiction** : saisie d'un dossier, score de risque et explication des facteurs par SHAP.
- **Scoring batch** : import d'un CSV, scoring de masse et export des résultats.
- **Impact business** : simulateur de scénarios pour arbitrer le seuil d'alerte.
- **Conformité RGPD** : minimisation des données, registre des traitements, journal d'audit.

## Déploiement sur Streamlit Community Cloud

1. Poussez le dépôt sur GitHub. Vérifiez que `.streamlit/secrets.toml` reste ignoré par Git.
2. Sur [share.streamlit.io](https://share.streamlit.io), créez une application pointant vers ce dépôt, avec `app.py` comme fichier principal.
3. Dans *Settings puis Secrets*, collez le bloc `[credentials]`.
4. Lancez le déploiement.

## Dépannage

**Le message « identifiant ou mot de passe incorrect » apparaît alors que les identifiants sont bons.**
Le fichier édité est sans doute `secrets.example.toml` et non `secrets.toml`. Vérifiez aussi que la commande est lancée depuis la racine du projet, car Streamlit cherche `.streamlit/secrets.toml` relativement au dossier courant.

**L'application ne trouve pas le modèle.**
Le fichier `models/xgb_optimise.joblib` doit être présent. Il est livré avec le dépôt ; s'il manque, récupérez-le depuis votre dernière version.

**Une colonne est signalée comme manquante lors d'un scoring par lot.**
Le CSV importé ne contient pas toutes les colonnes attendues. La liste exacte figure sur la page Scoring batch. La variable `assure_x_anticip` est recalculée automatiquement lorsque `est_assure_annulation` et `anticipation_jours` sont présentes.

## Conformité RGPD

Aucune donnée identifiante n'est utilisée par le modèle. Les variables telles que l'email, le nom, le téléphone, l'adresse et la date de naissance sont écartées dès la préparation des données. L'application n'affiche que des variables structurelles de réservation, exige une authentification et journalise les accès.

## Auteur

**Hady Coulibaly**, Master 2 Data et Intelligence Artificielle.

Projet académique. Les données réelles restent soumises à l'accord de confidentialité Maeva.
