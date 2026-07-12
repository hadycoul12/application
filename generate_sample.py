"""
Génère un sample_dataset.csv synthétique pour la démo.
À NE PAS utiliser en production — remplace par ton vrai dataset anonymisé.
Exécuter une seule fois : python generate_sample.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)

N = 2000

data = {
    "dossier_cle": [f"DOS_{i:06d}" for i in range(N)],
    "canal": np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.35, 0.20, 0.15, 0.12, 0.10, 0.08]),
    "est_assure_annulation": np.random.choice([0, 1], size=N, p=[0.55, 0.45]),
    "anticipation_jours": np.random.exponential(60, size=N).astype(int).clip(0, 365),
    "nb_dossiers_anterieurs": np.random.poisson(1.5, size=N).clip(0, 30),
    "est_solo": np.random.choice([0, 1], size=N, p=[0.65, 0.35]),
    "a_prestations": np.random.choice([0, 1], size=N, p=[0.60, 0.40]),
    "nb_produits_total": np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.45, 0.25, 0.15, 0.10, 0.05]),
    "region_destination": np.random.choice(range(15), size=N),
    "device_resa": np.random.choice([0, 1, 2], size=N, p=[0.55, 0.35, 0.10]),
    "theme_station": np.random.choice(range(10), size=N),
    "periode_depart": np.random.choice(range(1, 13), size=N),
    "duree_sejour": np.random.choice([2, 3, 5, 7, 14], size=N, p=[0.10, 0.15, 0.20, 0.40, 0.15]),
    "groupe_fournisseur": np.random.choice(range(8), size=N),
    "nb_campagnes_recues": np.random.poisson(3, size=N).clip(0, 50),
    "recence_email_jours": np.random.exponential(30, size=N).astype(int).clip(0, 365),
    "est_dans_crm": np.random.choice([0, 1], size=N, p=[0.90, 0.10]),
}

df = pd.DataFrame(data)

# Interaction dérivée
df["assure_x_anticip"] = df["est_assure_annulation"] * df["anticipation_jours"]

# Target synthétique (simulant ~7.3% d'annulation avec les bons gradients)
logit = (
    -3.5
    + 0.8 * (df["canal"] == 1).astype(float)       # Booking = plus d'annulation
    + 0.6 * df["est_assure_annulation"]              # Flex = plus d'annulation
    + 0.003 * df["anticipation_jours"]                # Plus d'anticipation = plus de risque
    + 0.1 * df["nb_dossiers_anterieurs"]              # Fidèles annulent plus
    + 0.15 * df["est_solo"]
    - 0.1 * df["nb_produits_total"]
    + 0.05 * df["a_prestations"]
    + np.random.normal(0, 0.3, N)
)
prob = 1 / (1 + np.exp(-logit))
df["y_annulation"] = (np.random.random(N) < prob).astype(int)

print(f"Dataset généré : {len(df)} lignes, taux annulation = {df['y_annulation'].mean()*100:.1f}%")

df.to_csv("data/sample_dataset.csv", index=False)
print("Fichier sauvegardé : data/sample_dataset.csv")
