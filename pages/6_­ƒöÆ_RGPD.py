"""
Page RGPD — Conformité, minimisation, journal d'audit, registre des traitements.
"""

import streamlit as st
import pandas as pd

from utils.auth import (
    check_rgpd_consent,
    check_authentication,
    log_action,
    get_current_user,
    get_audit_log,
)

# --- Contrôles d'accès ---
check_rgpd_consent()
check_authentication()

st.title("🔒 Conformité RGPD")
st.markdown(
    "Cette page documente les mesures de protection des données personnelles "
    "implémentées dans l'application."
)
st.markdown("---")

log_action(get_current_user(), "Accès page RGPD")

# ---------------------------------------------------------------------------
# 1. Consentement
# ---------------------------------------------------------------------------
st.markdown("### 1. Consentement")

consent_at = st.session_state.get("rgpd_consent_at", "Non enregistré")
st.markdown(
    f"""
    Une **bannière de consentement bloquante** est présentée à chaque utilisateur
    avant tout accès à l'application. Aucune donnée ne peut être traitée tant
    que le consentement n'a pas été explicitement donné.

    - **Consentement actuel :** ✅ Accepté
    - **Horodatage :** `{consent_at}`
    - **Stockage :** `session_state` (durée = session uniquement)
    - **Révocabilité :** déconnexion = révocation automatique
    """
)

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Minimisation des données
# ---------------------------------------------------------------------------
st.markdown("### 2. Minimisation des données (Art. 5 RGPD)")

st.markdown(
    "Le principe de minimisation impose de ne collecter que les données "
    "strictement nécessaires à la finalité du traitement. Les variables "
    "identifiantes suivantes ont été **exclues** du modèle et du dataset "
    "déployé :"
)

variables_exclues = pd.DataFrame(
    {
        "Variable exclue": [
            "client_email",
            "nom",
            "prenom",
            "telephone",
            "adresse_postale",
            "code_postal_client",
            "date_naissance",
        ],
        "Type": [
            "Email",
            "Identité",
            "Identité",
            "Contact",
            "Localisation",
            "Localisation",
            "Identité",
        ],
        "Justification": [
            "Donnée directement identifiante — supprimée dès l'export BigQuery",
            "Non nécessaire au scoring prédictif",
            "Non nécessaire au scoring prédictif",
            "Non nécessaire au scoring prédictif",
            "Non nécessaire — la région destination suffit",
            "Non nécessaire — la région destination suffit",
            "Non nécessaire au scoring prédictif",
        ],
    }
)

st.dataframe(variables_exclues, use_container_width=True, hide_index=True)

st.success(
    "✅ Seul l'identifiant interne `dossier_cle` est conservé comme clé "
    "de jointure. Cet identifiant n'est **pas réversible** vers le client "
    "sans accès à la base opérationnelle Maeva."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# 3. Sécurité
# ---------------------------------------------------------------------------
st.markdown("### 3. Mesures de sécurité")

securite = pd.DataFrame(
    {
        "Mesure": [
            "HTTPS",
            "Authentification",
            "Journal d'audit",
            "Anonymisation",
            "Pas de persistance",
            "Séparation des environnements",
        ],
        "Implémentation": [
            "Natif Streamlit Community Cloud — chiffrement en transit TLS 1.2+",
            "Login obligatoire via st.secrets — identifiants non stockés dans le code",
            "Chaque action sensible est horodatée (voir section ci-dessous)",
            "Aucune donnée personnelle identifiante dans le dataset déployé",
            "Prédictions en session uniquement — pas de base de données persistante",
            "Données d'entraînement sur PC sécurisé Maeva — seul le modèle est déployé",
        ],
    }
)

st.dataframe(securite, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 4. Journal d'audit
# ---------------------------------------------------------------------------
st.markdown("### 4. Journal des accès")

audit = get_audit_log()

if audit.empty:
    st.info("Aucune entrée dans le journal d'audit.")
else:
    st.markdown(f"**{len(audit)}** entrées enregistrées.")

    # Filtre par action
    actions_uniques = ["Toutes"] + audit["action"].unique().tolist()
    action_filter = st.selectbox("Filtrer par type d'action", actions_uniques)

    if action_filter != "Toutes":
        audit = audit[audit["action"] == action_filter]

    st.dataframe(
        audit.sort_index(ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Export
    csv_audit = audit.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        label="📥 Exporter le journal (CSV)",
        data=csv_audit,
        file_name="audit_log_export.csv",
        mime="text/csv",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# 5. Registre des traitements (Art. 30 RGPD)
# ---------------------------------------------------------------------------
st.markdown("### 5. Registre des traitements (Art. 30 RGPD)")

registre = pd.DataFrame(
    {
        "Champ": [
            "Responsable de traitement",
            "Finalité",
            "Base légale",
            "Catégories de données",
            "Catégories de personnes",
            "Destinataires",
            "Durée de conservation",
            "Mesures de sécurité",
            "Sous-traitants",
            "Transferts hors UE",
        ],
        "Description": [
            "Maeva / Groupe Pierre & Vacances Center Parcs",
            "Scoring prédictif du risque d'annulation de réservation — "
            "identification des dossiers à risque pour action de rétention ciblée",
            "Intérêt légitime (Art. 6.1.f) — optimisation de la gestion "
            "des réservations et réduction des pertes de CA",
            "Données de réservation anonymisées : canal, condition d'annulation, "
            "anticipation, durée, région, device, historique client (agrégé)",
            "Clients ayant effectué une réservation sur les plateformes Maeva",
            "Gestionnaires de réservation Maeva (accès authentifié)",
            "Session uniquement — aucune donnée personnelle persistée. "
            "Le modèle est entraîné hors ligne sur données anonymisées.",
            "HTTPS (TLS 1.2+), authentification, journalisation des accès, "
            "anonymisation des données, séparation entraînement/production",
            "Streamlit Community Cloud (hébergement USA) — données anonymisées "
            "uniquement, pas de données personnelles identifiantes",
            "Données anonymisées hébergées sur Streamlit Cloud (USA). "
            "Aucune donnée personnelle identifiante n'est transférée.",
        ],
    }
)

st.dataframe(registre, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 6. Mention AI Act (bonus)
# ---------------------------------------------------------------------------
st.markdown("### 6. Classification AI Act")

st.markdown(
    """
    Selon le Règlement européen sur l'Intelligence Artificielle (AI Act, 2024),
    ce système de scoring se classe en **risque limité** (catégorie non listée
    dans l'Annexe III des systèmes à haut risque).

    Le scoring porte sur des données de réservation commerciales anonymisées,
    sans impact sur les droits fondamentaux des personnes. L'obligation
    principale est la **transparence** : l'utilisateur est informé qu'il
    interagit avec un système de décision automatisée, et l'explicabilité
    est assurée par les explications SHAP.
    """
)

st.caption(
    "Cette page constitue une documentation de conformité à des fins "
    "pédagogiques dans le cadre d'un mémoire de Master 2. Elle ne remplace "
    "pas un audit RGPD mené par un DPO qualifié."
)
