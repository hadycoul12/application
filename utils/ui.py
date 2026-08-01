"""Composants d'interface réutilisables."""

import base64
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
CSS_PATH = ROOT / "assets" / "style.css"

# Emplacements et noms possibles pour le logo Maeva
_LOGO_CANDIDATES = [
    ROOT / "assets" / "logo_maeva.png",
    ROOT / "assets" / "logo_maeva.jpg",
    ROOT / "assets" / "logo_maeva.jpeg",
    ROOT / "assets" / "logo_maeva.svg",
    ROOT / "logo_maeva.png",
    ROOT / "logo_maeva.jpg",
    ROOT / "logo_maeva.jpeg",
]


@st.cache_data(show_spinner=False)
def find_logo() -> str | None:
    """Chemin du logo Maeva s'il existe, sinon None."""
    for p in _LOGO_CANDIDATES:
        if p.exists():
            return str(p)
    # Recherche souple : tout fichier commençant par « logo_maeva »
    for base in (ROOT / "assets", ROOT):
        if base.exists():
            for p in sorted(base.glob("logo_maeva*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
                    return str(p)
    return None


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str | None:
    """Logo encodé en data-URI, utilisable directement dans du HTML."""
    path = find_logo()
    if not path:
        return None
    p = Path(path)
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".webp": "image/webp",
    }.get(p.suffix.lower(), "image/png")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def inject_css():
    """Injecte la feuille de style globale."""
    try:
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass


def render_html(markup: str, *, sidebar: bool = False):
    """
    Rend du HTML dans Streamlit sans risque d'interprétation « bloc de code ».

    Streamlit traite tout bloc indenté de 4+ espaces comme du code : un HTML
    multi-ligne indenté (par confort de lecture dans le source) s'affiche alors
    en texte brut. On aplatit donc le HTML sur une seule ligne — sans indentation
    — avant le rendu. Les sauts de ligne deviennent des espaces (et non rien),
    pour ne pas coller entre eux les mots d'un texte réparti sur plusieurs lignes.
    """
    import re

    one_line = re.sub(r"\s*\n\s*", " ", markup.strip())  # sauts de ligne → espace
    one_line = re.sub(r">\s+<", "><", one_line)           # mais pas entre balises
    one_line = re.sub(r"\s{2,}", " ", one_line)           # espaces multiples → un
    target = st.sidebar if sidebar else st
    target.markdown(one_line, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "", show_logo: bool = False):
    """Bannière hero en dégradé. Affiche le logo Maeva si show_logo=True."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""

    logo_html = ""
    if show_logo:
        uri = logo_data_uri()
        if uri:
            logo_html = f'<img class="hero-logo" src="{uri}" alt="Maeva" />'

    html = (
        '<div class="hero">'
        f'{logo_html}'
        f'<div class="pill">{eyebrow}</div>'
        f'<h1>{title}</h1>'
        f'{sub}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section(label: str):
    """Titre de section avec filet accentué."""
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    """HTML d'une pastille. kind : ok | warn | danger | info | neutral"""
    return f'<span class="badge badge-{kind}">{text}</span>'


def sidebar_brand():
    """Bloc identité en tête de sidebar, avec logo si disponible."""
    uri = logo_data_uri()
    if uri:
        render_html(
            f'<div class="sb-brand">'
            f'<img class="sb-logo" src="{uri}" alt="Maeva" />'
            f'<div class="sub">Scoring d\'annulation</div>'
            f'</div>',
            sidebar=True,
        )
    else:
        render_html(
            '<div class="sb-brand">'
            '<div class="logo">🎯 Scoring Annulation</div>'
            '<div class="sub">Maeva · Pierre &amp; Vacances</div>'
            '</div>',
            sidebar=True,
        )


def sidebar_user(user: str):
    """Carte utilisateur connecté."""
    render_html(
        f'<div class="sb-user">'
        f'<div class="k">Session</div>'
        f'<div class="v">{user}</div>'
        f'<div class="tag">✓ Consentement RGPD validé</div>'
        f'</div>',
        sidebar=True,
    )
