import math
import random
import matplotlib.pyplot as plt

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matrix_multiply(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def copy_matrix(M):
    return [row[:] for row in M]

 
def compute_mean(data):
    n = len(data)
    dims = len(data[0])
    means = [0.0] * dims
    for row in data:
        for i, val in enumerate(row):
            means[i] += val
    means = [x / n for x in means]
    return means

def compute_std(data, means):
    n = len(data)
    dims = len(data[0])
    variances = [0.0] * dims
    for row in data:
        for i, val in enumerate(row):
            variances[i] += (val - means[i])**2
     
    stds = [math.sqrt(v / (n - 1)) for v in variances]
    return stds

def standardize_data(data):
    means = compute_mean(data)
    stds = compute_std(data, means)
    standardized = []
    for row in data:
        standardized.append([(val - mean) / std if std != 0 else 0 
                             for val, mean, std in zip(row, means, stds)])
    return standardized

 
def covariance_matrix(data):
    n = len(data)
    dims = len(data[0])
    cov = [[0.0] * dims for _ in range(dims)]
    for row in data:
        for i in range(dims):
            for j in range(dims):
                cov[i][j] += row[i] * row[j]
    for i in range(dims):
        for j in range(dims):
            cov[i][j] /= (n - 1)
    return cov

 
def max_offdiag(matrix):
    n = len(matrix)
    max_val = 0.0
    p = 0
    q = 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i][j]) > max_val:
                max_val = abs(matrix[i][j])
                p, q = i, j
    return p, q, max_val

def jacobi_eigen_decomposition(A, tol=1e-10, max_iterations=100):
    n = len(A)
    A = copy_matrix(A)   
    eigenvectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for iteration in range(max_iterations):
        p, q, max_val = max_offdiag(A)
        if max_val < tol:
            break   
        
        if A[p][p] == A[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
        
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        A_pp = A[p][p]
        A_qq = A[q][q]
        A_pq = A[p][q]
        A[p][p] = cos_t**2 * A_pp - 2*sin_t*cos_t*A_pq + sin_t**2 * A_qq
        A[q][q] = sin_t**2 * A_pp + 2*sin_t*cos_t*A_pq + cos_t**2 * A_qq
        A[p][q] = 0.0
        A[q][p] = 0.0
        
        for i in range(n):
            if i != p and i != q:
                A_ip = A[i][p]
                A_iq = A[i][q]
                A[i][p] = cos_t * A_ip - sin_t * A_iq
                A[p][i] = A[i][p]
                A[i][q] = sin_t * A_ip + cos_t * A_iq
                A[q][i] = A[i][q]
                
         
        for i in range(n):
            eigen_ip = eigenvectors[i][p]
            eigen_iq = eigenvectors[i][q]
            eigenvectors[i][p] = cos_t * eigen_ip - sin_t * eigen_iq
            eigenvectors[i][q] = sin_t * eigen_ip + cos_t * eigen_iq
            
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, eigenvectors

def sort_eigens(eigenvalues, eigenvectors):
    eig_pairs = [(eigenvalues[i], [vec[i] for vec in eigenvectors]) for i in range(len(eigenvalues))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    sorted_vals = [pair[0] for pair in eig_pairs]
    sorted_vecs = []
    for i in range(len(eigenvectors)):
        sorted_vecs.append([pair[1][i] for pair in eig_pairs])
    return sorted_vals, sorted_vecs

 

def perform_pca(data, n_components=None):
    standardized = standardize_data(data)
    cov_matrix = covariance_matrix(standardized)
    eigenvalues, eigenvectors = jacobi_eigen_decomposition(cov_matrix)
    eigenvalues, eigenvectors = sort_eigens(eigenvalues, eigenvectors)
    
    if n_components is not None:
        eigenvectors = [vec[:n_components] for vec in eigenvectors]
        eigenvalues = eigenvalues[:n_components]
    
    projected_data = []
    for row in standardized:
        projected_row = []
        for comp in range(len(eigenvalues)):
             
            dot = sum(row[i] * eigenvectors[i][comp] for i in range(len(row)))
            projected_row.append(dot)
        projected_data.append(projected_row)
    
    return projected_data, eigenvalues, eigenvectors

def generate_synthetic_data(n_samples=200):
    random.seed(42)
    data = []
    for _ in range(n_samples):
         
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        x = r * math.cos(theta)
         
        y = 0.5 * x + (r * math.sin(theta)) * 0.5
        data.append([x, y])
    return data

 

def main():
     
    data = generate_synthetic_data()
    projected_data, eigenvalues, eigenvectors = perform_pca(data, n_components=2)
    
    original_x = [pt[0] for pt in data]
    original_y = [pt[1] for pt in data]
    
    proj_x = [pt[0] for pt in projected_data]
    proj_y = [pt[1] for pt in projected_data]
    
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    
    axes[0].scatter(original_x, original_y, c='blue', alpha=0.5)
    axes[0].set_title('Original Synthetic Data')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
     
    axes[1].scatter(proj_x, proj_y, c='red', alpha=0.5)
    axes[1].set_title('PCA Projected Data')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
