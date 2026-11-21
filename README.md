# Scoring prédictif du risque d'annulation

> Application Streamlit qui estime, dès la réservation, la probabilité qu'un dossier soit annulé, explique pourquoi, et aide les équipes CRM à décider où concentrer leurs actions de rétention.

Projet réalisé pour mon mémoire de Master 2 Data & IA (Nexa Digital School), pendant mon alternance chez Maeva (groupe Pierre & Vacances Center Parcs).

## Le problème de départ

Une annulation tardive, c'est une chambre qui ne se revend pas et un chiffre d'affaires qui s'évapore. Les équipes savaient que certains dossiers étaient plus risqués que d'autres, mais sans moyen de les repérer à temps ni de savoir lesquels méritaient une relance.

Je voulais un outil simple à utiliser : on saisit (ou on importe) des dossiers, on obtient un score compréhensible, et on voit ce qui le fait monter ou descendre.

## Ce que fait l'application

| Page | À quoi elle sert |
|---|---|
| **Accueil** | Performances du modèle et diagnostic technique |
| **Dashboard** | Exploration du portefeuille : condition tarifaire, canal, saisonnalité, engagement e-mail |
| **Prédiction** | Score d'un dossier, avec l'explication SHAP des facteurs qui pèsent dans la décision |
| **Scoring batch** | Import d'un CSV, scoring de masse, export des résultats |
| **Impact business** | Simulation du chiffre d'affaires préservé selon le seuil d'alerte choisi |
| **Historique** | Journal des prédictions effectuées |
| **RGPD** | Minimisation des données, registre des traitements, journal d'audit |

## Résultats

Le modèle (XGBoost, probabilités recalibrées) a été évalué sur un portefeuille de **310 862 dossiers** :

- **AUC-ROC : 0,764**
- **PR-AUC : 0,263**, soit 3,6 fois mieux qu'un tirage au hasard (les annulations sont minoritaires, c'est la mesure la plus honnête ici)
- **Rappel : 72,7 %** : plus de 7 annulations sur 10 sont détectées avant qu'elles n'arrivent

Ces chiffres viennent du mémoire. Les détails de la démarche (analyse exploratoire, choix du modèle, tests de robustesse) sont dans le dépôt [`memoire_M2`](https://github.com/hadycoul12/memoire_M2).

## Quelques choix de conception

- **Rien n'est codé en dur.** Les listes de modalités, les champs du formulaire et les filtres sont lus dans le modèle entraîné et dans `data/schema.json`. Si les données changent, l'application suit.
- **Des scores qu'on peut expliquer.** Chaque prédiction s'accompagne de ses valeurs SHAP : un gestionnaire voit pourquoi un dossier est jugé risqué, ce qui compte pour qu'il fasse confiance à l'outil.
- **La confidentialité d'abord.** Aucune donnée identifiante (nom, e-mail, téléphone, adresse, date de naissance) n'entre dans le modèle. L'accès est protégé, consentement RGPD et journal d'audit compris.

## Limites, dites franchement

- Le signal issu des e-mails est une tendance, pas encore un résultat significatif : l'historique CRM ne couvrait que 7 mois. Avec 24 mois de données, la couverture pourrait passer à 40-50 %.
- Les données réelles de Maeva sont confidentielles et **ne sont pas publiées** dans ce dépôt. Le tableau de bord (qui s'appuie sur elles) n'est donc disponible que si vous fournissez votre propre jeu de données.
- Un accès de démonstration à l'application déployée est possible sur demande.

## Lancer le projet en local

Python 3.10 ou plus récent est nécessaire.

```bash
git clone https://github.com/hadycoul12/application.git
cd application
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

L'accès est protégé par un identifiant et un mot de passe que **vous choisissez**. Copiez le modèle de configuration, puis éditez la copie :

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Ce fichier `secrets.toml` est ignoré par Git. Ne le publiez jamais.

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Le modèle livré (`models/xgb_optimise.joblib`) permet d'utiliser les pages Prédiction et Scoring batch immédiatement.

### Utiliser vos propres données

Pour activer le tableau de bord et régénérer le schéma à partir de votre CSV :

```bash
python prepare_data.py "chemin/vers/votre_dataset.csv"
```

Les fichiers générés contenant des lignes de données restent en local. `setup_demo.py` permet aussi d'entraîner un modèle « jouet » pour tester l'application de bout en bout.

## Structure du projet

```
application/
├── app.py              Point d'entrée : consentement RGPD, connexion, navigation
├── prepare_data.py     Génère le schéma à partir d'un CSV
├── setup_demo.py       Entraîne un modèle de démonstration
├── views/              Les pages de l'application
├── utils/              Authentification, modèle, explications SHAP, interface
├── models/             Modèle et calibrateur (.joblib)
├── data/schema.json    Description des colonnes (aucune ligne de données)
└── assets/style.css    Feuille de style
```

## Déploiement sur Streamlit Community Cloud

1. Poussez le dépôt sur GitHub (le fichier `secrets.toml` doit rester ignoré).
2. Sur [share.streamlit.io](https://share.streamlit.io), créez une application pointant vers ce dépôt, fichier principal `app.py`.
3. Dans *Settings > Secrets*, collez le bloc `[credentials]` avec vos propres identifiants.

## Dépannage

**« Identifiant ou mot de passe incorrect » alors qu'ils sont bons.** Vérifiez que vous avez bien édité `secrets.toml` (et non `secrets.example.toml`) et que vous lancez la commande depuis la racine du projet.

**Une colonne est signalée comme manquante au scoring par lot.** Le CSV importé n'a pas toutes les colonnes attendues. La liste exacte est affichée sur la page Scoring batch. La variable `assure_x_anticip` est recalculée automatiquement.

## Auteur

**Hady Coulibaly**, data analyst et data scientist, statisticien de formation, titulaire d'un Master Data & IA.
[LinkedIn](https://www.linkedin.com/in/hady-coulibaly-data-ia-france-cdi/) · [Portfolio](https://portofolio-hadycoul.lovable.app/) · hady.coulibaly.data@gmail.com
