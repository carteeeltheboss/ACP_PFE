import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_s_curve
from matplotlib.patches import Ellipse
import seaborn as sns

def set_style():
    plt.style.use('classic')  # Use built-in matplotlib style
    sns.set_style("whitegrid")  # Apply seaborn grid style
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    # Add more seaborn-like parameters
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['grid.color'] = '0.8'
    plt.rcParams['grid.linestyle'] = '-'

def save_figure(name):
    plt.savefig(f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close()

# 1. Covariance Illustration
def plot_covariance():
    n_points = 200
    rng = np.random.RandomState(42)
    
    # Positive correlation
    cov_pos = [[1, 0.8], [0.8, 1]]
    data_pos = rng.multivariate_normal([0, 0], cov_pos, n_points)
    
    # Negative correlation
    cov_neg = [[1, -0.8], [-0.8, 1]]
    data_neg = rng.multivariate_normal([0, 0], cov_neg, n_points)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(data_pos[:, 0], data_pos[:, 1], alpha=0.5)
    ax1.set_title('Covariance Positive')
    ax2.scatter(data_neg[:, 0], data_neg[:, 1], alpha=0.5)
    ax2.set_title('Covariance Négative')
    
    save_figure('covariance_illustration')

# 2. Eigenvectors Visualization
def plot_eigenvectors():
    rng = np.random.RandomState(42)
    X = rng.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], 200)
    
    pca = PCA()
    pca.fit(X)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation
        plt.arrow(0, 0, comp[0], comp[1], 
                 color='r' if i == 0 else 'g',
                 width=0.05,
                 label=f'PC{i+1}')
    
    plt.axis('equal')
    plt.title('Vecteurs Propres de la Matrice de Covariance')
    plt.legend()
    save_figure('eigenvec')

# 3. Data Centering Visualization
def plot_data_centering():
    rng = np.random.RandomState(42)
    X = rng.multivariate_normal([3, 2], [[1, 0.5], [0.5, 1]], 200)
    
    X_centered = X - X.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax1.set_title('Données Originales')
    ax2.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5)
    ax2.set_title('Données Centrées')
    
    save_figure('centrage_normalisation')

# 4. Nonlinear PCA Failure
def plot_nonlinear_failure():
    X, _ = make_s_curve(1000, random_state=42)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='viridis')
    ax1.set_title('Données Originales (Projection 2D)')
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='b', alpha=0.5)
    ax2.set_title('Projection PCA')
    
    save_figure('nonlinear_pca_fail')

# 5. Outliers Impact
def plot_outliers_impact():
    rng = np.random.RandomState(42)
    X = rng.multivariate_normal([0, 0], [[2, 0.5], [0.5, 1]], 100)
    
    # Add outliers
    X_out = np.vstack([X, [[8, 8], [-8, -8]]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Without outliers
    pca = PCA()
    pca.fit(X)
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    for comp, var in zip(pca.components_, pca.explained_variance_):
        ax1.arrow(0, 0, comp[0]*var, comp[1]*var, color='r', width=0.05)
    ax1.set_title('Sans Valeurs Aberrantes')
    
    # With outliers
    pca.fit(X_out)
    ax2.scatter(X_out[:, 0], X_out[:, 1], alpha=0.5)
    for comp, var in zip(pca.components_, pca.explained_variance_):
        ax2.arrow(0, 0, comp[0]*var, comp[1]*var, color='r', width=0.05)
    ax2.set_title('Avec Valeurs Aberrantes')
    
    save_figure('outliers_impact')

# 6. Geometric Interpretation
def plot_geometric_interpretation():
    rng = np.random.RandomState(42)
    X = rng.multivariate_normal([0, 0], [[2, 1], [1, 2]], 200)
    pca = PCA()
    X_transformed = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax1.set_title('Espace Original')
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
    ax2.set_title('Après Rotation PCA')
    
    save_figure('pca_geometric')

if __name__ == "__main__":
    set_style()
    
    # Generate all figures
    plot_covariance()
    plot_eigenvectors()
    plot_data_centering()
    plot_nonlinear_failure()
    plot_outliers_impact()
    plot_geometric_interpretation()