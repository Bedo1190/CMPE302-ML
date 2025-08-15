import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Problem 1: PCA with Synthetic Dataset
X = np.array([
    [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
])

# (a) Standardize the data
mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - mean

# (b) Covariance matrix
cov_matrix = np.cov(X_centered)

# (c) Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# (d) Project onto the principal component with the largest eigenvalue
idx = np.argmax(eigenvalues)
principal_component = eigenvectors[:, idx].reshape(-1, 1)
X_projected = principal_component.T @ X_centered
print("Projected data (1D):", X_projected.flatten())

# Problem 2: PCA on Iris Dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris, cmap='viridis', edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris Dataset')
plt.grid(True)
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
