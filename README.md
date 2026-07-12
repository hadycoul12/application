# 🎯 Scoring prédictif — Risque d'annulation de réservation

Application web Streamlit de scoring prédictif des annulations de réservation, développée dans le cadre du mémoire de Master 2 Data & Intelligence Artificielle.

## 📋 Contexte

Ce projet implémente un modèle de Machine Learning (XGBoost) entraîné sur des données de réservation anonymisées pour identifier les dossiers à risque d'annulation. L'application permet aux gestionnaires de réservation de :

- **Visualiser** les tendances d'annulation (Dashboard)
- **Scorer** des dossiers individuels ou en batch
- **Comprendre** les facteurs de risque (explications SHAP)
- **Simuler** l'impact business d'une campagne de rétention ciblée
- **Respecter** la conformité RGPD (consentement, minimisation, audit)

## 🏗️ Architecture

```
scoring-annulation-maeva/
├── app.py                    # Point d'entrée (RGPD + Login + Accueil)
├── pages/
│   ├── 1_📊_Dashboard.py    # KPIs et visualisations
│   ├── 2_🎯_Prediction.py   # Scoring individuel + SHAP
│   ├── 3_📁_Batch.py        # Scoring batch (upload CSV)
│   ├── 4_💰_Impact_Business.py  # Simulation CA sauvé
│   ├── 5_📋_Historique.py   # Prédictions enregistrées
│   └── 6_🔒_RGPD.py         # Conformité et audit
├── utils/                    # Modules métier
├── models/                   # Modèle sérialisé (.joblib)
├── data/                     # Échantillon anonymisé
└── .streamlit/               # Configuration et thème
```

## 🚀 Installation locale

```bash
# Cloner le repo
git clone https://github.com/hadycoul12/scoring-annulation-maeva.git
cd scoring-annulation-maeva

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dépendances
pip install -r requirements.txt

# Configuration des secrets
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Éditer secrets.toml avec vos identifiants

# Générer le sample dataset (optionnel, pour la démo)
python generate_sample.py

# Lancer l'application
streamlit run app.py
```

## ☁️ Déploiement Streamlit Cloud

1. Pousser le repo sur GitHub
2. Se connecter à [share.streamlit.io](https://share.streamlit.io)
3. Sélectionner le repo et le fichier `app.py`
4. Configurer les secrets dans **Settings > Secrets** :
   ```toml
   [credentials]
   username = "manager@maeva.com"
   password = "votre_mot_de_passe"
   ```
5. Déployer

## 📊 Performances du modèle

| Métrique | Valeur |
|----------|--------|
| PR-AUC | 0.2626 (3.6× baseline) |
| Rappel | 62.7% |
| Précision | 16.8% |
| AUC-ROC | 0.7644 |
| Seuil retenu | 0.50 |

Modèle XGBoost optimisé par RandomizedSearchCV (100 itérations), pondération native des classes, sans SMOTE.

## 🔒 Conformité RGPD

- **Consentement** : bannière bloquante à l'entrée
- **Minimisation** : 7 variables identifiantes exclues (email, nom, prénom, téléphone, adresse, code postal, date de naissance)
- **Sécurité** : HTTPS natif, authentification obligatoire, journal d'audit horodaté
- **Anonymisation** : seul l'identifiant interne `dossier_cle` est conservé

## 📝 Licence

Projet académique — Mémoire Master 2 Data & IA, Nexa Digital School.

## 👤 Auteur

**Hady Coulibaly**
- LinkedIn : [linkedin.com/in/hady-coulibaly](https://linkedin.com/in/hady-coulibaly)
- GitHub : [github.com/hadycoul12](https://github.com/hadycoul12)
