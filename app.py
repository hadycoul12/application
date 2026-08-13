
import streamlit as st

st.set_page_config(
    page_title="Scoring Annulation · Maeva",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.auth import current_user, guard          # noqa: E402
from utils.ui import sidebar_brand, sidebar_user    # noqa: E402

#  Consentement RGPD puis authentification (bloquants) 
guard()

#  Sidebar 
sidebar_brand()
sidebar_user(current_user())

# Navigation 
pages = [
    st.Page("views/accueil.py",    title="Accueil",          icon=":material/home:", default=True),
    st.Page("views/dashboard.py",  title="Dashboard",        icon=":material/insights:"),
    st.Page("views/prediction.py", title="Prédiction",       icon=":material/target:"),
    st.Page("views/batch.py",      title="Scoring batch",    icon=":material/upload_file:"),
    st.Page("views/impact.py",     title="Impact business",  icon=":material/payments:"),
    st.Page("views/historique.py", title="Historique",       icon=":material/history:"),
    st.Page("views/rgpd.py",       title="Conformité RGPD",  icon=":material/shield:"),
]

nav = st.navigation(pages, position="sidebar")

# Pied de sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.link_button(
    "📖 Guide utilisateur (PDF)",
    "https://drive.google.com/file/d/1wVImy7cQ612LFl1JDMcz8Phsf4LRqqUy/view?usp=sharing",
    use_container_width=True,
)

if st.sidebar.button("Se déconnecter", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.caption("XGBoost calibré (isotonic) · PR-AUC 0.263")

nav.run()
