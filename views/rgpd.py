"""Conformité RGPD, consentement, minimisation, sécurité, registre, audit."""

import pandas as pd
import streamlit as st

from utils.auth import current_user, log, read_audit
from utils.ui import page_header, render_html, section

page_header(
    "Gouvernance",
    "Conformité RGPD",
    "Documentation des mesures de protection des données personnelles "
    "mises en œuvre dans cette application.",
)

log(current_user(), "Consultation de la page RGPD")

# ---------------------------------------------------------------------------
# Synthèse
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    render_html("""
        <div class="tile">
            <div class="icon">✅</div>
            <h4>Consentement</h4>
            <p>Bannière bloquante présentée à l'entrée. Aucun traitement
            n'est possible avant acceptation explicite.</p>
        </div>
        """)
with c2:
    render_html("""
        <div class="tile">
            <div class="icon">🗂️</div>
            <h4>Minimisation</h4>
            <p>7 variables directement identifiantes exclues du dataset
            et du modèle déployé.</p>
        </div>
        """)
with c3:
    render_html("""
        <div class="tile">
            <div class="icon">🔐</div>
            <h4>Sécurité</h4>
            <p>Chiffrement HTTPS, authentification obligatoire et
            journalisation horodatée des accès.</p>
        </div>
        """)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Minimisation", "Sécurité", "Registre des traitements", "Journal des accès"]
)

# ---------------------------------------------------------------------------
# 1. Minimisation
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("#### Principe de minimisation, article 5.1.c du RGPD")
    st.markdown(
        "Les données collectées doivent être « adéquates, pertinentes et limitées "
        "à ce qui est nécessaire au regard des finalités ». Les variables suivantes "
        "ont donc été **écartées dès l'export BigQuery**, avant toute modélisation."
    )

    exclues = pd.DataFrame(
        [
            ("client_email", "Identifiant direct", "Supprimée à l'export, aucune valeur prédictive une fois les features d'engagement agrégées"),
            ("nom", "Identité", "Sans lien causal avec le comportement d'annulation"),
            ("prenom", "Identité", "Sans lien causal avec le comportement d'annulation"),
            ("telephone", "Coordonnée", "Non nécessaire au scoring"),
            ("adresse_postale", "Localisation fine", "La région de destination suffit à capter l'effet géographique"),
            ("code_postal_client", "Localisation fine", "Risque de ré-identification par croisement ; granularité excessive"),
            ("date_naissance", "Identité", "Non nécessaire ; risque de biais discriminatoire lié à l'âge"),
        ],
        columns=["Variable exclue", "Catégorie", "Justification"],
    )
    st.dataframe(exclues, use_container_width=True, hide_index=True)

    st.success(
        "Seul l'identifiant interne `dossier_cle` est conservé comme clé technique. "
        "Il n'est pas réversible vers une personne physique sans accès à la base "
        "opérationnelle Maeva, hors périmètre de cette application.",
        icon=":material/verified_user:",
    )

# ---------------------------------------------------------------------------
# 2. Sécurité
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("#### Mesures techniques et organisationnelles, article 32 du RGPD")

    mesures = pd.DataFrame(
        [
            ("Chiffrement en transit", "HTTPS / TLS 1.2+ appliqué nativement par la plateforme d'hébergement", "Actif"),
            ("Contrôle d'accès", "Authentification obligatoire ; identifiants stockés hors du code source (st.secrets)", "Actif"),
            ("Journalisation", "Horodatage de chaque connexion, prédiction, export et purge", "Actif"),
            ("Anonymisation", "Aucune donnée personnelle identifiante dans le dataset déployé", "Actif"),
            ("Absence de persistance", "Prédictions conservées en session ; réinitialisation à chaque redéploiement", "Actif"),
            ("Séparation des environnements", "Entraînement sur poste sécurisé ; seul le modèle sérialisé est déployé", "Actif"),
        ],
        columns=["Mesure", "Implémentation", "Statut"],
    )
    st.dataframe(mesures, use_container_width=True, hide_index=True)

    st.markdown("#### Classification au titre de l'AI Act")
    st.markdown(
        """
        Le règlement européen sur l'intelligence artificielle (2024/1689) classe ce système
        en **risque limité** : il ne figure pas parmi les systèmes à haut risque de l'annexe III,
        n'affecte pas l'accès à un service essentiel et ne produit pas d'effet juridique
        sur les personnes.

        L'obligation principale est celle de **transparence** : l'utilisateur est informé
        qu'il consulte le résultat d'un système de décision automatisée, et l'explicabilité
        est assurée par les valeurs SHAP présentées à chaque prédiction.
        """
    )

# ---------------------------------------------------------------------------
# 3. Registre
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("#### Registre des activités de traitement, article 30 du RGPD")

    registre = pd.DataFrame(
        [
            ("Responsable de traitement", "Maeva, Groupe Pierre & Vacances Center Parcs"),
            ("Nom du traitement", "Scoring prédictif du risque d'annulation de réservation"),
            ("Finalité", "Identifier les réservations présentant un risque élevé d'annulation "
                         "afin de déclencher une action de rétention ciblée"),
            ("Base légale", "Intérêt légitime, article 6.1.f. L'optimisation de la gestion des "
                            "réservations ne porte pas atteinte aux droits des personnes concernées"),
            ("Catégories de données", "Données de réservation anonymisées : canal de distribution, "
                                      "condition tarifaire, délai d'anticipation, durée du séjour, "
                                      "région de destination, support de réservation, historique agrégé"),
            ("Catégories de personnes", "Clients ayant effectué une réservation sur les plateformes Maeva"),
            ("Destinataires", "Gestionnaires de réservation Maeva, accès authentifié"),
            ("Durée de conservation", "Session applicative uniquement. Les données d'entraînement sont "
                                      "conservées séparément selon la politique de rétention Maeva"),
            ("Transferts hors UE", "Hébergement Streamlit Community Cloud (États-Unis). "
                                   "Seules des données anonymisées y transitent, aucune donnée à "
                                   "caractère personnel n'est transférée"),
            ("Décision automatisée", "Le score constitue une aide à la décision. Aucune décision "
                                     "produisant un effet juridique n'est prise automatiquement"),
        ],
        columns=["Rubrique", "Contenu"],
    )
    st.dataframe(registre, use_container_width=True, hide_index=True, height=440)

# ---------------------------------------------------------------------------
# 4. Audit
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("#### Journal des accès et des traitements")

    audit = read_audit()

    if audit.empty:
        st.info("Aucune entrée enregistrée.", icon=":material/inbox:")
    else:
        consent_at = st.session_state.get("consent_at", "N/D")
        login_at = st.session_state.get("login_at", "N/D")

        a1, a2, a3 = st.columns(3)
        a1.metric("Entrées journalisées", f"{len(audit):,}".replace(",", " "))
        a2.metric("Consentement de session", consent_at)
        a3.metric("Connexion de session", login_at)

        st.write("")
        actions = ["Toutes"] + sorted(audit["action"].dropna().unique().tolist())
        pick = st.selectbox("Filtrer par action", actions)
        view = audit if pick == "Toutes" else audit[audit["action"] == pick]

        st.dataframe(view.iloc[::-1], use_container_width=True, hide_index=True, height=340)

        st.download_button(
            "Exporter le journal (CSV)",
            data=view.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="journal_audit.csv",
            mime="text/csv",
            icon=":material/download:",
        )

st.caption(
    "Cette documentation est produite dans un cadre académique (mémoire de Master 2). "
    "Elle ne se substitue pas à un audit mené par un Délégué à la Protection des Données."
)
