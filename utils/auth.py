"""Consentement RGPD, authentification et journal d'audit."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.ui import inject_css, render_html

AUDIT_LOG = Path(__file__).parent.parent / "data" / "audit_log.csv"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def _get_credentials() -> tuple[str, str] | None:
    """
    Récupère les identifiants depuis st.secrets.
    Retourne None si les secrets sont absents ou mal formés.
    """
    try:
        creds = st.secrets["credentials"]
        return str(creds["username"]).strip(), str(creds["password"]).strip()
    except Exception:
        return None


def _secrets_missing_screen():
    """Écran d'erreur explicite quand secrets.toml est introuvable."""
    st.error("⚠️ Configuration des identifiants introuvable")
    render_html("""
        Streamlit ne trouve pas la section `[credentials]` dans les secrets.

        **En local** — vérifie que le fichier `.streamlit/secrets.toml` existe
        (et non pas seulement `secrets.example.toml`), et que tu lances
        `streamlit run app.py` **depuis la racine du projet** :

        ```
        mon-projet/
        ├── .streamlit/
        │   └── secrets.toml     ← ce fichier exact
        └── app.py               ← lancer depuis ici
        ```

        Contenu attendu de `secrets.toml` :

        ```toml
        [credentials]
        username = "maeva"
        password = "maeva2026"
        ```

        **Sur Streamlit Cloud** — colle ce même contenu dans
        *Settings → Secrets*, puis relance l'application.
        """
    )
    st.stop()


# ---------------------------------------------------------------------------
# Consentement RGPD
# ---------------------------------------------------------------------------
def require_consent():
    """Bannière de consentement bloquante."""
    if st.session_state.get("consent"):
        return

    inject_css()

    render_html("""
        <div class="gate">
            <div class="gate-hero">
                <div class="shield">🛡️</div>
                <h2>Protection de vos données personnelles</h2>
                <p>
                    Conformément au Règlement Général sur la Protection des Données
                    (RGPD — UE 2016/679), nous vous informons du traitement mis en œuvre
                    dans cette application.
                </p>
            </div>
            <div class="gate-body">
                <div class="gate-row">
                    <div class="k">🎯 Finalité</div>
                    <div class="v">Identifier les réservations présentant un risque élevé
                        d'annulation, en vue d'une action de rétention ciblée.</div>
                </div>
                <div class="gate-row">
                    <div class="k">⚖️ Base légale</div>
                    <div class="v">Intérêt légitime — article 6.1.f du RGPD.</div>
                </div>
                <div class="gate-row">
                    <div class="k">🗂️ Données traitées</div>
                    <div class="v">Variables structurelles de réservation uniquement :
                        canal, condition tarifaire, anticipation, durée, région, support.
                        <b>Aucune donnée personnelle identifiante</b> — les variables
                        nom, email, téléphone, adresse et date de naissance sont exclues.</div>
                </div>
                <div class="gate-row">
                    <div class="k">🔒 Sécurité</div>
                    <div class="v">Transmission chiffrée HTTPS, authentification obligatoire,
                        journalisation horodatée des accès.</div>
                </div>
                <div class="gate-row">
                    <div class="k">⏱️ Conservation</div>
                    <div class="v">Session uniquement — aucune persistance au-delà
                        de la déconnexion.</div>
                </div>
                <div class="gate-row">
                    <div class="k">✋ Vos droits</div>
                    <div class="v">Accès, rectification, effacement, portabilité, opposition.
                        Contact DPO : hady.coulibaly@edu.nexa.fr — réponse sous 30 jours.</div>
                </div>
            </div>
        </div>
        """)

    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        lu = st.checkbox(
            "J'ai lu et j'accepte le traitement de ces données. Je comprends mes droits RGPD."
        )
        if st.button("Accéder à l'application →", type="primary", use_container_width=True):
            if not lu:
                st.warning("Vous devez cocher la case pour accéder à l'application.")
            else:
                st.session_state["consent"] = True
                st.session_state["consent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log("anonyme", "Consentement RGPD accepté")
                st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
def require_login():
    """Écran de connexion bloquant."""
    if st.session_state.get("auth"):
        return

    inject_css()

    creds = _get_credentials()
    if creds is None:
        _secrets_missing_screen()

    valid_user, valid_pass = creds

    render_html('<div class="login-wrap">')
    render_html("""
        <div class="login-logo">🎯</div>
        <div class="login-title">Scoring Annulation</div>
        <div class="login-sub">Accès réservé aux gestionnaires de réservation</div>
        """)

    with st.form("login", border=True):
        user = st.text_input("Identifiant", placeholder="maeva")
        pwd = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        ok = st.form_submit_button("Se connecter", type="primary", use_container_width=True)

    if ok:
        # .strip() côté saisie : évite les échecs dus à un espace collé
        if user.strip() == valid_user and pwd.strip() == valid_pass:
            st.session_state["auth"] = True
            st.session_state["user"] = user.strip()
            st.session_state["login_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(user.strip(), "Connexion réussie")
            st.rerun()
        else:
            log(user.strip() or "inconnu", "Échec de connexion")
            st.error("Identifiant ou mot de passe incorrect.")

    render_html("</div>")
    st.stop()


# ---------------------------------------------------------------------------
# Journal d'audit
# ---------------------------------------------------------------------------
def log(user: str, action: str):
    """Enregistre une action horodatée."""
    row = pd.DataFrame(
        [{
            "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "utilisateur": user,
            "action": action,
        }]
    )
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        header = not AUDIT_LOG.exists()
        row.to_csv(AUDIT_LOG, mode="a", header=header, index=False, encoding="utf-8")
    except Exception:
        pass  # ne jamais bloquer l'app sur une erreur d'écriture


def read_audit() -> pd.DataFrame:
    """Charge le journal d'audit."""
    if AUDIT_LOG.exists():
        try:
            return pd.read_csv(AUDIT_LOG, encoding="utf-8")
        except Exception:
            pass
    return pd.DataFrame(columns=["horodatage", "utilisateur", "action"])


def current_user() -> str:
    return st.session_state.get("user", "inconnu")


def guard():
    """Garde unique à appeler en haut de chaque vue."""
    inject_css()
    require_consent()
    require_login()
