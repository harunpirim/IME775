import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Week 1: Mathematical Foundations - Linear Algebra for Deep Learning

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapters 1-2

    ---

    ## Learning Objectives

    - Understand vectors, matrices, and tensors as data representations
    - Master matrix operations essential for neural networks
    - Connect linear algebra to neural network computations
    - Visualize transformations geometrically
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Vectors: The Building Blocks

    ### The Dot Product: Heart of Neural Networks

    The dot product is the fundamental operation in neural networks:
    $$\mathbf{w}^T \mathbf{x} = \sum_{i=1}^{n} w_i x_i$$

    **Geometric interpretation**: Measures alignment between vectors
    $$\mathbf{w}^T \mathbf{x} = \|\mathbf{w}\| \|\mathbf{x}\| \cos\theta$$
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Visualize dot product as projection
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Vector alignment
    ax1 = axes1[0]
    w = np.array([3, 1])
    x1 = np.array([2, 2])
    x2 = np.array([1, -2])

    ax1.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', label=f'w = {w}', width=0.02)
    ax1.quiver(0, 0, x1[0], x1[1], angles='xy', scale_units='xy', scale=1, 
               color='green', label=f'x₁ = {x1}, w·x₁ = {np.dot(w, x1)}', width=0.02)
    ax1.quiver(0, 0, x2[0], x2[1], angles='xy', scale_units='xy', scale=1, 
               color='red', label=f'x₂ = {x2}, w·x₂ = {np.dot(w, x2)}', width=0.02)

    ax1.set_xlim(-3, 4)
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('Dot Product: Measures Alignment')

    # Right: Neuron as dot product
    ax2 = axes1[1]
    angles = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(angles), np.sin(angles), 'b-', alpha=0.3)

    # Sample points and their dot products with w
    np.random.seed(42)
    points = np.random.randn(50, 2) * 0.8
    w_normalized = w / np.linalg.norm(w)
    dots = points @ w_normalized

    scatter = ax2.scatter(points[:, 0], points[:, 1], c=dots, cmap='RdYlGn', 
                          s=60, edgecolors='black', linewidth=0.5)
    ax2.quiver(0, 0, w_normalized[0], w_normalized[1], angles='xy', 
               scale_units='xy', scale=1, color='blue', width=0.02, label='Weight direction')
    plt.colorbar(scatter, ax=ax2, label='Dot product value')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title('Neuron Response: w·x for Different Inputs')
    ax2.legend()

    plt.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.2 Matrix Operations in Neural Networks

    ### Matrix-Vector Multiplication: Layer Transformation

    $$\mathbf{y} = \mathbf{W}\mathbf{x}$$

    Each output is a dot product of a weight row with input:
    $$y_i = \mathbf{w}_i^T \mathbf{x}$$

    **Neural network interpretation**:
    - $\mathbf{x} \in \mathbb{R}^n$: Input activations
    - $\mathbf{W} \in \mathbb{R}^{m \times n}$: Weight matrix
    - $\mathbf{y} \in \mathbb{R}^m$: Output (before activation)
    """)
    return


@app.cell
def _(np, plt):
    # Visualize matrix transformation
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    # Original points (circle)
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    # Different transformations
    transformations = [
        (np.array([[2, 0], [0, 1]]), 'Scaling: diag(2, 1)'),
        (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], 
                   [np.sin(np.pi/4), np.cos(np.pi/4)]]), 'Rotation: 45°'),
        (np.array([[1, 0.5], [0.5, 1]]), 'Shear + Scale')
    ]

    for ax, (W, title) in zip(axes2, transformations):
        # Original circle
        ax.plot(circle[0], circle[1], 'b-', alpha=0.3, linewidth=2, label='Original')

        # Transformed circle
        transformed = W @ circle
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')

        # Show basis vectors
        ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
                  color='blue', alpha=0.5, width=0.02)
        ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
                  color='blue', alpha=0.5, width=0.02)
        ax.quiver(0, 0, W[0, 0], W[1, 0], angles='xy', scale_units='xy', scale=1, 
                  color='red', alpha=0.5, width=0.02)
        ax.quiver(0, 0, W[0, 1], W[1, 1], angles='xy', scale_units='xy', scale=1, 
                  color='red', alpha=0.5, width=0.02)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.3 Eigenvalues and Eigenvectors

    ### Definition
    For a square matrix $\mathbf{A}$:
    $$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

    Eigenvectors define directions that are only **scaled** (not rotated) by the transformation.

    ### Eigendecomposition
    For symmetric $\mathbf{A}$:
    $$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$
    """)
    return


@app.cell
def _(np, plt):
    # Visualize eigenvectors
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Symmetric matrix
    A = np.array([[3, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Left: Eigenvalue/eigenvector visualization
    ax1_eigen = axes3[0]

    # Draw transformed circle
    theta_e = np.linspace(0, 2*np.pi, 100)
    circle_e = np.vstack([np.cos(theta_e), np.sin(theta_e)])
    transformed_e = A @ circle_e

    ax1_eigen.plot(circle_e[0], circle_e[1], 'b-', alpha=0.3, linewidth=2, label='Original circle')
    ax1_eigen.plot(transformed_e[0], transformed_e[1], 'r-', linewidth=2, label='Transformed')

    # Draw eigenvectors
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        ax1_eigen.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                        color=['green', 'purple'][i], width=0.03, 
                        label=f'v{i+1}: λ={val:.2f}')
        ax1_eigen.quiver(0, 0, val*vec[0], val*vec[1], angles='xy', scale_units='xy', scale=1,
                        color=['green', 'purple'][i], alpha=0.3, width=0.02)

    ax1_eigen.set_xlim(-4, 4)
    ax1_eigen.set_ylim(-4, 4)
    ax1_eigen.set_aspect('equal')
    ax1_eigen.grid(True, alpha=0.3)
    ax1_eigen.legend()
    ax1_eigen.set_title(f'Eigenvectors of A = [[3,1],[1,2]]')

    # Right: PCA interpretation
    ax2_eigen = axes3[1]
    np.random.seed(42)

    # Generate correlated data
    cov = np.array([[2, 1.5], [1.5, 2]])
    data = np.random.multivariate_normal([0, 0], cov, 200)

    # Compute PCA via eigendecomposition
    data_centered = data - data.mean(axis=0)
    cov_matrix = np.cov(data_centered.T)
    pca_eigenvalues, pca_eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue (descending)
    idx = np.argsort(pca_eigenvalues)[::-1]
    pca_eigenvalues = pca_eigenvalues[idx]
    pca_eigenvectors = pca_eigenvectors[:, idx]

    ax2_eigen.scatter(data[:, 0], data[:, 1], alpha=0.4, s=20)

    for i, (val, vec) in enumerate(zip(pca_eigenvalues, pca_eigenvectors.T)):
        scale = np.sqrt(val) * 2
        ax2_eigen.quiver(0, 0, vec[0]*scale, vec[1]*scale, angles='xy', 
                        scale_units='xy', scale=1, color=['red', 'blue'][i],
                        width=0.03, label=f'PC{i+1}: λ={val:.2f}')

    ax2_eigen.set_xlim(-5, 5)
    ax2_eigen.set_ylim(-5, 5)
    ax2_eigen.set_aspect('equal')
    ax2_eigen.grid(True, alpha=0.3)
    ax2_eigen.legend()
    ax2_eigen.set_title('PCA: Eigenvectors of Covariance Matrix')

    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.4 Singular Value Decomposition (SVD)

    ### Definition
    Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed:
    $$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

    ### Low-Rank Approximation
    Truncated SVD with top $k$ singular values provides the best rank-$k$ approximation.

    **Applications**: Model compression, dimensionality reduction
    """)
    return


@app.cell
def _(np, plt):
    # SVD for image compression
    from PIL import Image
    import urllib.request
    import io

    # Create a simple test image (gradient pattern)
    def create_test_image(size=100):
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        img = np.sin(5*x) * np.cos(5*y) + 0.5*np.sin(10*x + 5*y)
        img = (img - img.min()) / (img.max() - img.min())
        return img

    image = create_test_image(200)

    # Compute SVD
    U_img, sigma_img, Vt_img = np.linalg.svd(image, full_matrices=False)

    # Reconstruct with different numbers of singular values
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))

    ks = [1, 5, 10, 25, 50, 200]
    for ax, k in zip(axes4.flat, ks):
        reconstructed = U_img[:, :k] @ np.diag(sigma_img[:k]) @ Vt_img[:k, :]
        ax.imshow(reconstructed, cmap='gray')

        # Compression ratio
        original_size = image.size
        compressed_size = k * (U_img.shape[0] + Vt_img.shape[1] + 1)
        compression = compressed_size / original_size * 100

        error = np.linalg.norm(image - reconstructed, 'fro') / np.linalg.norm(image, 'fro')
        ax.set_title(f'k={k}, Error={error:.3f}\nStorage: {compression:.1f}%')
        ax.axis('off')

    plt.suptitle('SVD Image Compression: Low-Rank Approximation', fontsize=14, y=1.02)
    plt.tight_layout()
    fig4
    return (sigma_img,)


@app.cell
def _(np, plt, sigma_img):
    # Singular value spectrum
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 4))

    ax1_svd = axes5[0]
    ax1_svd.semilogy(sigma_img, 'b.-')
    ax1_svd.set_xlabel('Index')
    ax1_svd.set_ylabel('Singular Value (log scale)')
    ax1_svd.set_title('Singular Value Spectrum')
    ax1_svd.grid(True, alpha=0.3)

    ax2_svd = axes5[1]
    cumulative_energy = np.cumsum(sigma_img**2) / np.sum(sigma_img**2)
    ax2_svd.plot(cumulative_energy, 'g.-')
    ax2_svd.axhline(0.9, color='r', linestyle='--', label='90% energy')
    ax2_svd.axhline(0.99, color='orange', linestyle='--', label='99% energy')
    k_90 = np.searchsorted(cumulative_energy, 0.9) + 1
    k_99 = np.searchsorted(cumulative_energy, 0.99) + 1
    ax2_svd.axvline(k_90, color='r', linestyle=':', alpha=0.5)
    ax2_svd.axvline(k_99, color='orange', linestyle=':', alpha=0.5)
    ax2_svd.set_xlabel('Number of Singular Values (k)')
    ax2_svd.set_ylabel('Cumulative Energy')
    ax2_svd.set_title(f'Energy Captured: 90% at k={k_90}, 99% at k={k_99}')
    ax2_svd.legend()
    ax2_svd.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.5 Norms and Regularization

    | Norm | Definition | Deep Learning Use |
    |------|------------|-------------------|
    | L1 | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Sparsity (Lasso) |
    | L2 | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Weight decay |
    | Frobenius | $\|\mathbf{A}\|_F = \sqrt{\sum_{ij} a_{ij}^2}$ | Matrix regularization |
    """)
    return


@app.cell
def _(np, plt):
    # Visualize norm balls
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4))

    theta_norm = np.linspace(0, 2*np.pi, 1000)

    # L1 ball
    ax1_norm = axes6[0]
    l1_ball = np.vstack([np.sign(np.cos(theta_norm)) * np.abs(np.cos(theta_norm)),
                         np.sign(np.sin(theta_norm)) * np.abs(np.sin(theta_norm))])
    # Diamond shape
    t = np.linspace(0, 2*np.pi, 5)
    l1_x = np.cos(t)
    l1_y = np.sin(t)
    l1_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax1_norm.plot(l1_points[:, 0], l1_points[:, 1], 'b-', linewidth=2)
    ax1_norm.fill(l1_points[:, 0], l1_points[:, 1], alpha=0.3)
    ax1_norm.set_title('L1 Ball: $\|x\|_1 \leq 1$\nPromotes Sparsity')
    ax1_norm.set_xlim(-1.5, 1.5)
    ax1_norm.set_ylim(-1.5, 1.5)
    ax1_norm.set_aspect('equal')
    ax1_norm.grid(True, alpha=0.3)

    # L2 ball
    ax2_norm = axes6[1]
    l2_x = np.cos(theta_norm)
    l2_y = np.sin(theta_norm)
    ax2_norm.plot(l2_x, l2_y, 'g-', linewidth=2)
    ax2_norm.fill(l2_x, l2_y, alpha=0.3, color='green')
    ax2_norm.set_title('L2 Ball: $\|x\|_2 \leq 1$\nWeight Decay')
    ax2_norm.set_xlim(-1.5, 1.5)
    ax2_norm.set_ylim(-1.5, 1.5)
    ax2_norm.set_aspect('equal')
    ax2_norm.grid(True, alpha=0.3)

    # L-inf ball
    ax3_norm = axes6[2]
    linf_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    ax3_norm.plot(linf_points[:, 0], linf_points[:, 1], 'r-', linewidth=2)
    ax3_norm.fill(linf_points[:, 0], linf_points[:, 1], alpha=0.3, color='red')
    ax3_norm.set_title('L∞ Ball: $\|x\|_∞ \leq 1$\nAdversarial Robustness')
    ax3_norm.set_xlim(-1.5, 1.5)
    ax3_norm.set_ylim(-1.5, 1.5)
    ax3_norm.set_aspect('equal')
    ax3_norm.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Definition | Neural Network Role |
    |---------|------------|---------------------|
    | **Dot Product** | $\mathbf{w}^T\mathbf{x}$ | Core neuron computation |
    | **Matrix Mult.** | $\mathbf{Y} = \mathbf{WX}$ | Layer transformation |
    | **Eigendecomp.** | $\mathbf{A} = \mathbf{Q\Lambda Q}^T$ | PCA, Hessian analysis |
    | **SVD** | $\mathbf{A} = \mathbf{U\Sigma V}^T$ | Compression, init. |
    | **Norms** | $\|\cdot\|_p$ | Regularization |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapters 1-2.
    - **Supplementary**: Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 2.

    ## Connection to ML Refined Curriculum

    This linear algebra foundation supports:
    - Week 1: Understanding feature representations
    - Weeks 2-3: Gradient computations
    - Week 8: PCA as eigendecomposition
    """)
    return


if __name__ == "__main__":
    app.run()
