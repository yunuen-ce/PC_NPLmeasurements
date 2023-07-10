import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate a synthetic dataset with 4 features and 300 samples
n_samples = 300
n_features = 4
X, y = make_blobs(n_samples=n_samples, centers=4, n_features=n_features, random_state=42, cluster_std=2)

# Apply PCA to reduce the dimensionality to 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Visualize the transformed 3D data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolors='k')
ax.set_title("PCA Transformed 3D Data")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
