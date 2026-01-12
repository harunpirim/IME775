import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Week 8: Linear Unsupervised Learning & PCA

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 8**

    ---

    ## Learning Objectives

    - Understand unsupervised learning concepts
    - Implement Principal Component Analysis (PCA)
    - Apply K-Means clustering
    - Introduction to matrix factorization
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    return KMeans, PCA, load_iris, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction (Section 8.1)

    **Unsupervised Learning**: Learn patterns from data without labels.

    ### Key Problems

    | Problem | Goal |
    |---------|------|
    | **Dimensionality Reduction** | Compress data to fewer dimensions |
    | **Clustering** | Group similar data points |
    | **Anomaly Detection** | Find unusual points |
    | **Matrix Factorization** | Decompose data matrix |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Principal Component Analysis (Section 8.3)

    ### Goal

    Find a low-dimensional representation that captures most of the variance.

    ### The Linear Autoencoder View

    Encode data $x$ into low-dimensional $z$, then decode back:
    - **Encode**: $z = C^T x$ where $C \in \mathbb{R}^{n \times k}$
    - **Decode**: $\hat{x} = C z$

    ### Optimization Problem

    $$\min_C \frac{1}{P} \sum_{p=1}^{P} \|x_p - CC^T x_p\|^2$$

    Subject to: $C^T C = I$ (orthonormal columns)

    ### Solution

    The optimal $C$ consists of the top $k$ eigenvectors of the covariance matrix:
    $$\Sigma = \frac{1}{P} \sum_{p=1}^{P} (x_p - \bar{x})(x_p - \bar{x})^T$$
    """)
    return


@app.cell
def _(PCA, load_iris, np, plt):
    # PCA on Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize
    X_centered = X - X.mean(axis=0)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_centered)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Explained variance
    ax1 = axes[0]
    ax1.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, label='Individual')
    ax1.plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='Cumulative')
    ax1.axhline(0.95, color='g', linestyle='--', label='95% threshold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot (ML Refined, Section 8.3)')
    ax1.legend()
    ax1.set_xticks(range(1, 5))
    
    # 2D projection
    ax2 = axes[1]
    colors = ['blue', 'orange', 'green']
    for c in range(3):
        mask = y == c
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], 
                   label=iris.target_names[c], s=50, alpha=0.7)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Iris Dataset: 2D PCA Projection')
    ax2.legend()
    
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## K-Means Clustering (Section 8.5)

    ### Goal

    Partition data into $K$ clusters to minimize within-cluster variance.

    ### Cost Function

    $$g(c, \mu) = \sum_{k=1}^{K} \sum_{p: c_p = k} \|x_p - \mu_k\|^2$$

    Where:
    - $c_p \in \{1, \ldots, K\}$: Cluster assignment for point $p$
    - $\mu_k$: Centroid of cluster $k$

    ### Algorithm

    ```
    1. Initialize: Random centroids μ₁, ..., μₖ
    2. Repeat until convergence:
       a. Assign: cₚ = argminₖ ||xₚ - μₖ||²
       b. Update: μₖ = mean of points in cluster k
    ```
    """)
    return


@app.cell
def _(KMeans, X, np, plt, y):
    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # True labels
    ax1 = axes2[0]
    colors = ['blue', 'orange', 'green']
    for c in range(3):
        mask = y == c
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=50, alpha=0.7)
    ax1.set_xlabel('Sepal Length')
    ax1.set_ylabel('Sepal Width')
    ax1.set_title('True Labels')
    
    # K-Means clusters
    ax2 = axes2[1]
    for c in range(3):
        mask = clusters == c
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=50, alpha=0.7)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax2.set_xlabel('Sepal Length')
    ax2.set_ylabel('Sepal Width')
    ax2.set_title('K-Means Clusters')
    ax2.legend()
    
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recommender Systems (Section 8.4)

    ### Matrix Factorization

    Approximate rating matrix $R \in \mathbb{R}^{M \times N}$ as:
    $$R \approx UV^T$$

    Where:
    - $U \in \mathbb{R}^{M \times k}$: User features
    - $V \in \mathbb{R}^{N \times k}$: Item features
    - $k$: Number of latent factors

    ### Cost Function

    $$g(U, V) = \sum_{(i,j) \in \Omega} (R_{ij} - u_i^T v_j)^2 + \lambda(\|U\|^2 + \|V\|^2)$$

    Where $\Omega$ is the set of observed ratings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Goal | Key Idea |
    |--------|------|----------|
    | **PCA** | Dimension reduction | Find directions of maximum variance |
    | **K-Means** | Clustering | Minimize within-cluster variance |
    | **Matrix Factorization** | Recommendations | Decompose into latent factors |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 8.
    - **Supplementary**: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 12.

    ## Next Week

    **Feature Engineering and Selection** (Chapter 9): Preparing data for ML models.
    """)
    return


if __name__ == "__main__":
    app.run()
