import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)

# Figure 1: Covariance illustration
def plot_covariance_illustration():
    # Generate two datasets with different correlations
    n_points = 200
    
    # Positive correlation
    cov_pos = [[1, 0.8], [0.8, 1]]
    data_pos = np.random.multivariate_normal([0, 0], cov_pos, n_points)
    
    # Negative correlation
    cov_neg = [[1, -0.8], [-0.8, 1]]
    data_neg = np.random.multivariate_normal([0, 0], cov_neg, n_points)
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot positive correlation
    ax1.scatter(data_pos[:, 0], data_pos[:, 1], alpha=0.5)
    ax1.set_title('Covariance Positive')
    ax1.set_xlabel('Variable X')
    ax1.set_ylabel('Variable Y')
    
    # Plot negative correlation
    ax2.scatter(data_neg[:, 0], data_neg[:, 1], alpha=0.5)
    ax2.set_title('Covariance Négative')
    ax2.set_xlabel('Variable X')
    ax2.set_ylabel('Variable Y')
    
    plt.tight_layout()
    plt.savefig('covariance_illustration.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2: PCA geometric interpretation
def plot_pca_geometric():
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
    
    # Rotate data to make it more interesting
    theta = np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    X = X @ rotation_matrix
    
    # Fit PCA
    pca = PCA()
    pca.fit(X)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Données')
    
    # Plot principal components
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp_line = comp * var  # scale component by its variance explanation
        plt.arrow(0, 0, comp_line[0], comp_line[1], 
                 color='r' if i == 0 else 'g',
                 width=0.05,
                 label=f'PC{i+1}')
    
    plt.axis('equal')
    plt.title('Interprétation Géométrique de l\'ACP')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('pca_geometric.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate both figures
    plot_covariance_illustration()
    plot_pca_geometric()