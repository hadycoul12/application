"""Composants d'interface réutilisables."""

from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"


def inject_css():
    """Injecte la feuille de style globale."""
    try:
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    """Bannière hero en dégradé."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">{eyebrow}</div>
            <h1>{title}</h1>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str):
    """Titre de section avec filet accentué."""
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    """HTML d'une pastille. kind : ok | warn | danger | info | neutral"""
    return f'<span class="badge badge-{kind}">{text}</span>'


def sidebar_brand():
    """Bloc identité en tête de sidebar."""
    st.sidebar.markdown(
        """
        <div class="sb-brand">
            <div class="logo">🎯 Scoring Annulation</div>
            <div class="sub">Maeva · Pierre &amp; Vacances</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_user(user: str):
    """Carte utilisateur connecté."""
    st.sidebar.markdown(
        f"""
        <div class="sb-user">
            <div class="k">Session</div>
            <div class="v">{user}</div>
            <div class="tag">✓ Consentement RGPD validé</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
