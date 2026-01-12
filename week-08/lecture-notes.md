# Week 8: Linear Unsupervised Learning & PCA

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 8: Linear Unsupervised Learning**.

---

## Overview

This week covers unsupervised learning methods including PCA for dimensionality reduction and K-Means for clustering.

---

## Learning Objectives

- Understand unsupervised learning concepts
- Derive and implement Principal Component Analysis (PCA)
- Apply K-Means clustering
- Introduction to matrix factorization for recommender systems

---

## 8.1 Introduction

### Unsupervised Learning

Learn patterns from data without labels:
- No $y$ values, only $\{x_1, \ldots, x_P\}$
- Find hidden structure

### Key Applications

| Application | Method |
|-------------|--------|
| Visualization | Dimensionality reduction |
| Data compression | PCA, autoencoders |
| Customer segmentation | Clustering |
| Recommendations | Matrix factorization |
| Anomaly detection | Density estimation |

---

## 8.2 Fixed Spanning Sets, Orthonormality, and Projections

### Orthonormal Basis

Vectors $\{c_1, \ldots, c_k\}$ are orthonormal if:
$$c_i^T c_j = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

### Projection

The projection of $x$ onto the subspace spanned by orthonormal $C = [c_1, \ldots, c_k]$:
$$\hat{x} = CC^T x$$

### Reconstruction Error

$$\|x - \hat{x}\|^2 = \|x - CC^T x\|^2$$

---

## 8.3 The Linear Autoencoder and Principal Component Analysis

### The Autoencoder View

- **Encoder**: $z = C^T x$ (compress to $k$ dimensions)
- **Decoder**: $\hat{x} = Cz$ (reconstruct)

### PCA Optimization Problem

$$\min_{C: C^T C = I} \frac{1}{P} \sum_{p=1}^{P} \|x_p - CC^T x_p\|^2$$

Equivalently (maximize variance captured):
$$\max_{C: C^T C = I} \frac{1}{P} \sum_{p=1}^{P} \|C^T x_p\|^2$$

### Solution

The optimal $C$ consists of the top $k$ eigenvectors of the covariance matrix:

$$\Sigma = \frac{1}{P} X^T X$$

where $X$ is the centered data matrix.

### Algorithm

```python
def pca(X, k):
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Covariance matrix
    cov = X_centered.T @ X_centered / len(X)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k
    C = eigenvectors[:, :k]
    
    # Project data
    Z = X_centered @ C
    
    return Z, C, eigenvalues
```

### Choosing $k$

1. **Scree plot**: Look for "elbow"
2. **Variance threshold**: Keep components explaining 95% variance
3. **Kaiser criterion**: Keep eigenvalues > 1 (for standardized data)

### Explained Variance Ratio

$$\text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

---

## 8.4 Recommender Systems

### Problem

Predict missing entries in a rating matrix.

### Matrix Factorization

Approximate $R \approx UV^T$ where:
- $R \in \mathbb{R}^{M \times N}$: Rating matrix (users × items)
- $U \in \mathbb{R}^{M \times k}$: User latent factors
- $V \in \mathbb{R}^{N \times k}$: Item latent factors

### Cost Function

$$g(U, V) = \sum_{(i,j) \in \Omega} (R_{ij} - u_i^T v_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

### Alternating Least Squares

Fix $V$, optimize $U$; then fix $U$, optimize $V$.

---

## 8.5 K-Means Clustering

### Goal

Partition $\{x_1, \ldots, x_P\}$ into $K$ clusters.

### Cost Function

$$g(c, \mu) = \sum_{k=1}^{K} \sum_{p: c_p = k} \|x_p - \mu_k\|^2$$

### Lloyd's Algorithm

```python
def kmeans(X, K, max_iter=100):
    # Random initialization
    centroids = X[np.random.choice(len(X), K, replace=False)]
    
    for _ in range(max_iter):
        # Assign points to nearest centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[assignments == k].mean(axis=0) 
                                  for k in range(K)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return assignments, centroids
```

### Initialization Strategies

- **Random**: Simple but may converge to local minimum
- **K-Means++**: Smart initialization, spread out centroids
- **Multiple restarts**: Run several times, keep best

### Choosing $K$

- **Elbow method**: Plot cost vs $K$, look for elbow
- **Silhouette score**: Measure cluster cohesion vs separation
- **Domain knowledge**: Prior information about number of groups

---

## 8.6 General Matrix Factorization Techniques

### Non-negative Matrix Factorization (NMF)

$$\min_{U, V \geq 0} \|X - UV^T\|_F^2$$

Constraint: All entries of $U$ and $V$ are non-negative.

Applications:
- Topic modeling
- Image decomposition
- Audio source separation

### Sparse Coding

$$\min_{D, Z} \|X - DZ\|_F^2 + \lambda \|Z\|_1$$

Learn dictionary $D$ and sparse codes $Z$.

---

## Exercises

### Exercise 8.1 (Section 8.3)
Implement PCA from scratch. Compare with sklearn's PCA on the Iris dataset.

### Exercise 8.2 (Section 8.5)
Run K-Means with different initializations. How does the final cost vary?

### Exercise 8.3 (Section 8.3-8.5)
Apply PCA for visualization before clustering. Does it improve interpretability?

---

## Summary

- PCA finds directions of maximum variance
- K-Means minimizes within-cluster variance
- Matrix factorization decomposes data into latent factors
- All methods optimize interpretable cost functions

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 8**.

### Supplementary Reading
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 12.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapters 11-12.

---

## Next Week Preview

**Week 9: Feature Engineering and Selection** (Chapter 9)
- Histogram features
- Feature scaling
- Feature selection via boosting and regularization
