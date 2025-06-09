import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from matplotlib.ticker import PercentFormatter

def generate_scree_plot():
    # Generate sample data
    n_samples = 1000
    n_features = 10
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)

    # Perform PCA
    pca = PCA()
    pca.fit(X)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Scree plot with elbow criterion
    components = range(1, len(explained_variance_ratio) + 1)
    ax1.plot(components, explained_variance_ratio, 'bo-', linewidth=2)
    ax1.set_xlabel('Composante Principale')
    ax1.set_ylabel('Variance Expliquée')
    ax1.set_title('Critère du Coude')
    ax1.grid(True)

    # Add elbow annotation
    elbow_point = 3  # This is typically determined visually or algorithmically
    ax1.annotate('Point du coude',
                xy=(elbow_point, explained_variance_ratio[elbow_point-1]),
                xytext=(elbow_point+1, explained_variance_ratio[elbow_point-1]+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Plot 2: Cumulative variance with threshold lines
    ax2.plot(components, cumulative_variance_ratio, 'ro-', linewidth=2)
    ax2.axhline(y=0.8, color='g', linestyle='--', label='Seuil 80%')
    ax2.axhline(y=0.9, color='b', linestyle='--', label='Seuil 90%')
    ax2.set_xlabel('Nombre de Composantes')
    ax2.set_ylabel('Variance Cumulée Expliquée')
    ax2.set_title('Critère de la Variance Cumulée')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(True)
    ax2.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('scree_plot_criteria.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_scree_plot()