import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Supposons que vous avez téléchargé un fichier CSV contenant les données de température de surface
# Exemple : 'tao_data.csv' avec des colonnes pour la date, la latitude, la longitude et la température

# Chargement des données
df = pd.read_csv('tao_data.csv', parse_dates=['date'])

# Filtrage des données pour une période spécifique, par exemple 1980-2020
df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '2020-12-31')]

# Pivotement des données pour avoir les dates en lignes et les emplacements (lat, lon) en colonnes
df_pivot = df.pivot_table(index='date', columns=['latitude', 'longitude'], values='temperature')

# Gestion des valeurs manquantes
df_pivot = df_pivot.interpolate().dropna()

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pivot)

# Application de l'ACP
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5)
plt.xlabel(f'Composante principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Composante principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('ACP des températures de surface - Données TAO/TRITON')
plt.grid(True)
plt.show()
