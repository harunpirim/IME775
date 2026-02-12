"""
IME 775: PCA and Singular Value Decomposition
==============================================
A marimo notebook exploring PCA, SVD, and their applications
in dimensionality reduction and document retrieval.

Course: IME 775 - Mathematical Foundations of Deep Learning
Topics: PCA, SVD, Dimensionality Reduction, LSA
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # IME 775: PCA and Singular Value Decomposition

        ## Learning Objectives

        1. Understand PCA as finding directions of maximum variance
        2. Visualize dimensionality reduction via projection
        3. Connect PCA to eigendecomposition of covariance matrix
        4. Master SVD and its applications (PCA, solving systems, low-rank approximation)

        ---
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return np, plt, Axes3D


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. The Core Idea of PCA

        **Goal:** Find the direction(s) along which data varies the most.

        **Why?**
        - Data often lies near a lower-dimensional subspace
        - Projecting onto this subspace preserves most information
        - Reduces noise (small variations perpendicular to pattern)
        - Compresses data for storage/computation

        **Mathematical Foundation:**
        - Variance along direction $\hat{l}$: $\sigma^2 = \hat{l}^T C \hat{l}$ (quadratic form!)
        - Maximum variance occurs along eigenvector of $C$ with largest eigenvalue
        """
    )
    return


@app.cell
def __(mo):
    # PCA demo parameters
    n_samples = mo.ui.slider(50, 300, value=150, step=10, label="Number of samples")
    spread_major = mo.ui.slider(1, 5, value=3, step=0.25, label="Spread along major axis")
    spread_minor = mo.ui.slider(0.1, 2, value=0.5, step=0.1, label="Spread along minor axis")
    rotation_deg = mo.ui.slider(0, 90, value=30, step=5, label="Data rotation (degrees)")

    mo.md(f"""
    ### Interactive PCA Demo

    Generate 2D data with controlled spread and orientation:

    {n_samples}
    {spread_major}
    {spread_minor}
    {rotation_deg}
    """)
    return n_samples, spread_major, spread_minor, rotation_deg


@app.cell
def __(np, plt, n_samples, spread_major, spread_minor, rotation_deg, mo):
    np.random.seed(42)

    # Generate data
    n = n_samples.value
    sigma1 = spread_major.value
    sigma2 = spread_minor.value
    theta = np.radians(rotation_deg.value)

    # Create elongated data
    X_raw = np.column_stack([
        sigma1 * np.random.randn(n),
        sigma2 * np.random.randn(n)
    ])

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Rotate and shift
    X = (R @ X_raw.T).T
    X += np.array([2, 1])  # Add offset

    # PCA computation
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Covariance matrix
    C = np.cov(X_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var * 100

    # Project onto PC1
    pc1 = eigenvectors[:, 0]
    projections = X_centered @ pc1
    X_reconstructed_1d = np.outer(projections, pc1) + mean

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Original data with principal components
    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, s=30, c='blue', label='Data')
    ax1.scatter(mean[0], mean[1], s=200, c='red', marker='x', linewidths=3, label='Mean')

    # Draw principal components
    scale = 2
    for i in range(2):
        ev = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
        color = 'green' if i == 0 else 'orange'
        ax1.quiver(mean[0], mean[1], ev[0], ev[1], angles='xy', scale_units='xy', scale=1,
                   color=color, width=0.03, label=f'PC{i+1} ({var_explained[i]:.1f}%)')
        ax1.quiver(mean[0], mean[1], -ev[0], -ev[1], angles='xy', scale_units='xy', scale=1,
                   color=color, width=0.03, alpha=0.5)

    ax1.set_xlabel('$x_0$')
    ax1.set_ylabel('$x_1$')
    ax1.set_title('Original Data + Principal Components')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Middle: Projection onto PC1
    ax2 = axes[1]
    ax2.scatter(X[:, 0], X[:, 1], alpha=0.3, s=30, c='blue', label='Original')
    ax2.scatter(X_reconstructed_1d[:, 0], X_reconstructed_1d[:, 1], alpha=0.7, s=30, c='red', label='Projected')

    # Draw projection lines (subset for clarity)
    for i in range(0, n, max(1, n//20)):
        ax2.plot([X[i, 0], X_reconstructed_1d[i, 0]],
                 [X[i, 1], X_reconstructed_1d[i, 1]], 'gray', alpha=0.3)

    # Draw PC1 line
    t = np.linspace(-4, 4, 100)
    pc1_line = mean[:, np.newaxis] + np.outer(pc1, t)
    ax2.plot(pc1_line[0], pc1_line[1], 'g-', linewidth=2, label='PC1 (projection line)')

    ax2.set_xlabel('$x_0$')
    ax2.set_ylabel('$x_1$')
    ax2.set_title('Dimensionality Reduction: 2D → 1D')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Right: Variance explained
    ax3 = axes[2]
    ax3.bar([1, 2], var_explained, color=['green', 'orange'], alpha=0.7)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['PC1', 'PC2'])
    ax3.set_ylabel('Variance Explained (%)')
    ax3.set_title('Variance Explained by Components')
    ax3.set_ylim(0, 100)

    # Add cumulative line
    cumulative = np.cumsum(var_explained)
    ax3_twin = ax3.twinx()
    ax3_twin.plot([1, 2], cumulative, 'ro-', linewidth=2, label='Cumulative')
    ax3_twin.set_ylabel('Cumulative %')
    ax3_twin.set_ylim(0, 105)
    ax3_twin.legend(loc='right')

    plt.tight_layout()

    # Compute reconstruction error
    reconstruction_error = np.mean(np.sum((X - X_reconstructed_1d)**2, axis=1))

    mo.md(f"""
    ### PCA Results

    **Covariance Matrix:**
    ```
    C = [{C[0,0]:7.3f}  {C[0,1]:7.3f}]
        [{C[1,0]:7.3f}  {C[1,1]:7.3f}]
    ```

    **Principal Components:**

    | PC | Eigenvalue | Variance Explained | Cumulative |
    |----|------------|-------------------|------------|
    | PC1 | {eigenvalues[0]:.3f} | {var_explained[0]:.1f}% | {var_explained[0]:.1f}% |
    | PC2 | {eigenvalues[1]:.3f} | {var_explained[1]:.1f}% | {cumulative[1]:.1f}% |

    **Reconstruction Error (using PC1 only):** {reconstruction_error:.4f}

    **Key Insight:** Most variance is along PC1 (the elongation direction). Projecting onto PC1 preserves {var_explained[0]:.1f}% of the information!
    """)
    return n, sigma1, sigma2, theta, X_raw, R, X, mean, X_centered, C, eigenvalues, eigenvectors, var_explained, pc1, projections, X_reconstructed_1d, fig, reconstruction_error


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 2. PCA in 3D → 2D Reduction

        When data clusters around a plane in 3D, PCA reveals this by showing:
        - Two large eigenvalues (spread within the plane)
        - One small eigenvalue (spread normal to plane)

        Discarding the smallest principal component projects data onto the best-fit plane.
        """
    )
    return


@app.cell
def __(mo):
    # 3D PCA parameters
    n_3d = mo.ui.slider(100, 500, value=200, step=50, label="Number of 3D points")
    noise_3d = mo.ui.slider(0.1, 2, value=0.5, step=0.1, label="Out-of-plane noise")
    plane_angle = mo.ui.slider(0, 90, value=45, step=5, label="Plane tilt angle")

    mo.md(f"""
    ### 3D to 2D Dimensionality Reduction

    {n_3d}
    {noise_3d}
    {plane_angle}
    """)
    return n_3d, noise_3d, plane_angle


@app.cell
def __(np, plt, n_3d, noise_3d, plane_angle, Axes3D, mo):
    np.random.seed(123)

    # Generate data on a tilted plane + noise
    n3 = n_3d.value
    tilt = np.radians(plane_angle.value)

    # Points on XY plane
    X_plane = np.random.randn(n3, 2) * 2

    # Tilt around X axis
    R_tilt = np.array([[1, 0, 0],
                       [0, np.cos(tilt), -np.sin(tilt)],
                       [0, np.sin(tilt), np.cos(tilt)]])

    # Create 3D points (on tilted plane + noise perpendicular to plane)
    X_3d_raw = np.column_stack([X_plane[:, 0], X_plane[:, 1], np.zeros(n3)])
    X_3d = (R_tilt @ X_3d_raw.T).T

    # Add noise perpendicular to plane
    normal = R_tilt @ np.array([0, 0, 1])
    X_3d += noise_3d.value * np.outer(np.random.randn(n3), normal)

    # Center
    mean_3d = np.mean(X_3d, axis=0)
    X_3d_centered = X_3d - mean_3d

    # PCA
    C_3d = np.cov(X_3d_centered.T)
    evals_3d, evecs_3d = np.linalg.eigh(C_3d)
    idx_3d = np.argsort(evals_3d)[::-1]
    evals_3d = evals_3d[idx_3d]
    evecs_3d = evecs_3d[:, idx_3d]

    var_exp_3d = evals_3d / np.sum(evals_3d) * 100

    # Project onto first 2 PCs
    W_2d = evecs_3d[:, :2]
    X_projected_2d = X_3d_centered @ W_2d

    # Reconstruct
    X_reconstructed_3d = X_projected_2d @ W_2d.T + mean_3d

    # Visualization
    fig2 = plt.figure(figsize=(16, 5))

    # 3D scatter
    ax1 = fig2.add_subplot(131, projection='3d')
    ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.5, s=20, c='blue')

    # Draw principal axes
    scale_3d = 2
    colors_3d = ['green', 'orange', 'red']
    for i in range(3):
        ev = evecs_3d[:, i] * np.sqrt(evals_3d[i]) * scale_3d
        ax1.quiver(mean_3d[0], mean_3d[1], mean_3d[2],
                   ev[0], ev[1], ev[2], color=colors_3d[i], linewidth=2,
                   label=f'PC{i+1} ({var_exp_3d[i]:.1f}%)')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Data with Principal Axes')
    ax1.legend()

    # 2D projection
    ax2 = fig2.add_subplot(132)
    ax2.scatter(X_projected_2d[:, 0], X_projected_2d[:, 1], alpha=0.5, s=20, c='purple')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'Projected to 2D\n({var_exp_3d[0]+var_exp_3d[1]:.1f}% variance retained)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Variance bar chart
    ax3 = fig2.add_subplot(133)
    bars = ax3.bar([1, 2, 3], var_exp_3d, color=colors_3d, alpha=0.7)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['PC1', 'PC2', 'PC3'])
    ax3.set_ylabel('Variance Explained (%)')
    ax3.set_title('Variance per Component')
    ax3.axhline(y=var_exp_3d[2], color='red', linestyle='--',
                label=f'PC3: {var_exp_3d[2]:.1f}% (discard)')
    ax3.legend()

    plt.tight_layout()

    recon_err_3d = np.mean(np.sum((X_3d - X_reconstructed_3d)**2, axis=1))

    mo.md(f"""
    ### 3D PCA Results

    **Eigenvalues:** λ₁ = {evals_3d[0]:.2f}, λ₂ = {evals_3d[1]:.2f}, λ₃ = {evals_3d[2]:.2f}

    **Variance Explained:** PC1: {var_exp_3d[0]:.1f}%, PC2: {var_exp_3d[1]:.1f}%, PC3: {var_exp_3d[2]:.1f}%

    **Retained (2 PCs):** {var_exp_3d[0]+var_exp_3d[1]:.1f}%

    **Reconstruction Error:** {recon_err_3d:.4f}

    **Insight:** PC3 captures only {var_exp_3d[2]:.1f}% variance (the out-of-plane noise). Discarding it projects data onto the underlying plane with minimal information loss!
    """)
    return n3, tilt, X_plane, R_tilt, X_3d_raw, X_3d, normal, mean_3d, X_3d_centered, C_3d, evals_3d, evecs_3d, var_exp_3d, W_2d, X_projected_2d, X_reconstructed_3d, fig2, recon_err_3d


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 3. Singular Value Decomposition (SVD)

        **Theorem:** Any matrix $A \in \mathbb{R}^{m \times n}$ can be written as:

        $$A = U \Sigma V^T$$

        where:
        - $U$ ($m \times m$): orthogonal, columns are eigenvectors of $AA^T$
        - $\Sigma$ ($m \times n$): diagonal with singular values $\sigma_i = \sqrt{\lambda_i(A^TA)}$
        - $V$ ($n \times n$): orthogonal, columns are eigenvectors of $A^TA$

        **Key Properties:**
        - Works for ANY matrix (not just square or symmetric)
        - Singular values are always non-negative
        - Provides orthonormal bases for row and column spaces
        """
    )
    return


@app.cell
def __(mo):
    # SVD visualization
    svd_rows = mo.ui.slider(2, 5, value=3, step=1, label="Number of rows (m)")
    svd_cols = mo.ui.slider(2, 5, value=4, step=1, label="Number of columns (n)")
    svd_seed = mo.ui.slider(1, 100, value=42, step=1, label="Random seed")

    mo.md(f"""
    ### SVD Visualization

    Create a random matrix and visualize its SVD:

    {svd_rows}
    {svd_cols}
    {svd_seed}
    """)
    return svd_rows, svd_cols, svd_seed


@app.cell
def __(np, plt, svd_rows, svd_cols, svd_seed, mo):
    np.random.seed(svd_seed.value)

    m = svd_rows.value
    n_svd = svd_cols.value

    # Create random matrix
    A_svd = np.random.randn(m, n_svd)

    # SVD
    U_svd, S_svd, Vt_svd = np.linalg.svd(A_svd, full_matrices=True)

    # Create Sigma matrix
    Sigma = np.zeros((m, n_svd))
    for i in range(min(m, n_svd)):
        Sigma[i, i] = S_svd[i]

    # Verify reconstruction
    A_reconstructed = U_svd @ Sigma @ Vt_svd
    reconstruction_err = np.linalg.norm(A_svd - A_reconstructed)

    # Visualization
    fig3, axes = plt.subplots(1, 5, figsize=(18, 4))

    # Plot each matrix
    matrices = [A_svd, U_svd, Sigma, Vt_svd, A_reconstructed]
    titles = ['A', 'U', 'Σ', 'Vᵀ', 'UΣVᵀ']
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set_title(f'{title}\n{mat.shape}', fontsize=12)
        # Add values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'SVD: A = UΣVᵀ  (reconstruction error: {reconstruction_err:.2e})', fontsize=14)
    plt.tight_layout()

    mo.md(f"""
    ### SVD Components

    **Singular Values:** {', '.join([f'σ{i+1}={s:.3f}' for i, s in enumerate(S_svd)])}

    **Matrix Dimensions:**
    - A: {m} × {n_svd}
    - U: {m} × {m} (orthogonal)
    - Σ: {m} × {n_svd} (diagonal)
    - Vᵀ: {n_svd} × {n_svd} (orthogonal)

    **Verification:**
    - $U^T U = I$: {np.allclose(U_svd.T @ U_svd, np.eye(m))}
    - $V V^T = I$: {np.allclose(Vt_svd @ Vt_svd.T, np.eye(n_svd))}
    - $A = U\\Sigma V^T$: error = {reconstruction_err:.2e}
    """)
    return m, n_svd, A_svd, U_svd, S_svd, Vt_svd, Sigma, A_reconstructed, reconstruction_err, fig3


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 4. Low-Rank Approximation

        **Goal:** Approximate matrix $A$ (rank $p$) with a rank-$r$ matrix ($r < p$).

        **Best Approximation (Eckart-Young Theorem):**

        $$A_r = \sum_{i=1}^{r} \sigma_i \vec{u}_i \vec{v}_i^T$$

        Keep only the $r$ largest singular values and corresponding vectors.

        **Approximation Error:**
        $$\|A - A_r\|_F = \sqrt{\sum_{i=r+1}^{p} \sigma_i^2}$$
        """
    )
    return


@app.cell
def __(mo):
    rank_approx = mo.ui.slider(1, 4, value=1, step=1, label="Rank of approximation (r)")

    mo.md(f"""
    ### Low-Rank Approximation Demo

    Using the matrix from above, approximate with rank:

    {rank_approx}
    """)
    return rank_approx,


@app.cell
def __(np, plt, A_svd, U_svd, S_svd, Vt_svd, rank_approx, mo):
    r = min(rank_approx.value, len(S_svd))

    # Low-rank approximation
    A_lowrank = np.zeros_like(A_svd)
    for i in range(r):
        A_lowrank += S_svd[i] * np.outer(U_svd[:, i], Vt_svd[i, :])

    # Errors
    approx_error = np.linalg.norm(A_svd - A_lowrank, 'fro')
    original_norm = np.linalg.norm(A_svd, 'fro')
    relative_error = approx_error / original_norm * 100

    # Theoretical error
    theoretical_error = np.sqrt(np.sum(S_svd[r:]**2))

    # Energy retained
    energy_retained = np.sum(S_svd[:r]**2) / np.sum(S_svd**2) * 100

    # Visualization
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    # Original
    ax1 = axes4[0]
    im1 = ax1.imshow(A_svd, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax1.set_title(f'Original A\n(rank {len(S_svd)})')
    plt.colorbar(im1, ax=ax1)

    # Approximation
    ax2 = axes4[1]
    im2 = ax2.imshow(A_lowrank, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax2.set_title(f'Rank-{r} Approximation\n({energy_retained:.1f}% energy)')
    plt.colorbar(im2, ax=ax2)

    # Error
    ax3 = axes4[2]
    im3 = ax3.imshow(A_svd - A_lowrank, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax3.set_title(f'Error: A - A_r\n(‖·‖_F = {approx_error:.3f})')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()

    mo.md(f"""
    ### Approximation Results

    | Metric | Value |
    |--------|-------|
    | Rank of approximation | **{r}** |
    | Frobenius error | **{approx_error:.4f}** |
    | Theoretical error | **{theoretical_error:.4f}** |
    | Relative error | **{relative_error:.1f}%** |
    | Energy retained | **{energy_retained:.1f}%** |

    **Singular values used:** {', '.join([f'σ{i+1}={S_svd[i]:.3f}' for i in range(r)])}

    **Singular values discarded:** {', '.join([f'σ{i+1}={S_svd[i]:.3f}' for i in range(r, len(S_svd))])}
    """)
    return r, A_lowrank, approx_error, original_norm, relative_error, theoretical_error, energy_retained, fig4


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 5. Document Retrieval with LSA

        **Problem:** Standard TF-IDF + cosine similarity only finds exact term matches.

        **Solution:** Latent Semantic Analysis (LSA) uses SVD to:
        1. Find "topics" (linear combinations of terms that co-occur)
        2. Project documents into topic space
        3. Measure similarity in topic space

        Documents sharing topics are similar even without shared exact terms!
        """
    )
    return


@app.cell
def __(np, plt, mo):
    # Document-term matrix (rows=docs, cols=terms)
    # Terms: violence, gun, america, roses
    terms = ["violence", "gun", "america", "roses"]
    docs = ["d0: Roses are lovely", "d1: Gun violence epidemic", "d2: Gun violence issue",
            "d3: Guns beget violence", "d4: I like guns, hate violence", "d5: Gun robbery",
            "d6: Acts of violence"]

    # Term frequencies (simplified)
    X_docs = np.array([
        [0, 0, 0, 2],  # d0: roses only
        [1, 1, 1, 0],  # d1: violence, gun, america
        [2, 2, 0, 0],  # d2: violence, gun
        [3, 3, 0, 0],  # d3: violence, gun
        [5, 5, 0, 0],  # d4: violence, gun (many times)
        [0, 1, 0, 0],  # d5: gun only
        [1, 0, 0, 0],  # d6: violence only
    ], dtype=float)

    # SVD
    U_docs, S_docs, Vt_docs = np.linalg.svd(X_docs, full_matrices=False)

    # Cosine similarity function
    def cosine_sim(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        return np.dot(v1, v2) / (n1 * n2)

    # Direct cosine similarity (d5 vs d6)
    direct_sim_5_6 = cosine_sim(X_docs[5], X_docs[6])

    # LSA: project to topic space (keep top 2 topics)
    V_topics = Vt_docs.T[:, :2]  # First 2 topics
    X_topic_space = X_docs @ V_topics

    # LSA similarity
    lsa_sim_5_6 = cosine_sim(X_topic_space[5], X_topic_space[6])

    # Visualization
    fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))

    # Document-term matrix
    ax1 = axes5[0]
    im1 = ax1.imshow(X_docs, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(terms)))
    ax1.set_xticklabels(terms, rotation=45)
    ax1.set_yticks(range(len(docs)))
    ax1.set_yticklabels([f'd{i}' for i in range(len(docs))])
    ax1.set_title('Document-Term Matrix')
    for i in range(X_docs.shape[0]):
        for j in range(X_docs.shape[1]):
            ax1.text(j, i, f'{X_docs[i,j]:.0f}', ha='center', va='center')
    plt.colorbar(im1, ax=ax1)

    # Topic vectors
    ax2 = axes5[1]
    width = 0.35
    x_pos = np.arange(len(terms))
    ax2.bar(x_pos - width/2, V_topics[:, 0], width, label='Topic 1', alpha=0.7)
    ax2.bar(x_pos + width/2, V_topics[:, 1], width, label='Topic 2', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(terms, rotation=45)
    ax2.set_ylabel('Weight')
    ax2.set_title('Topic Vectors (from V)')
    ax2.legend()
    ax2.axhline(y=0, color='k', linewidth=0.5)

    # Documents in topic space
    ax3 = axes5[2]
    ax3.scatter(X_topic_space[:, 0], X_topic_space[:, 1], s=100, c='blue', alpha=0.7)
    for i, doc in enumerate(docs):
        ax3.annotate(f'd{i}', (X_topic_space[i, 0], X_topic_space[i, 1]),
                     xytext=(5, 5), textcoords='offset points')

    # Highlight d5 and d6
    ax3.scatter([X_topic_space[5, 0]], [X_topic_space[5, 1]], s=200, c='red', marker='s', label='d5')
    ax3.scatter([X_topic_space[6, 0]], [X_topic_space[6, 1]], s=200, c='green', marker='^', label='d6')
    ax3.plot([X_topic_space[5, 0], X_topic_space[6, 0]],
             [X_topic_space[5, 1], X_topic_space[6, 1]], 'k--', alpha=0.5)

    ax3.set_xlabel('Topic 1 (gun-violence)')
    ax3.set_ylabel('Topic 2')
    ax3.set_title('Documents in Topic Space')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md(f"""
    ### LSA Results: d5 vs d6 Similarity

    **d5:** "Guns were used in an armed robbery" (contains "gun")
    **d6:** "Acts of violence usually involve a weapon" (contains "violence")

    | Method | Similarity |
    |--------|------------|
    | Direct TF-IDF cosine | **{direct_sim_5_6:.4f}** |
    | LSA (topic space) | **{lsa_sim_5_6:.4f}** |

    **Explanation:** Direct comparison finds zero overlap (no shared terms). But LSA recognizes that "gun" and "violence" frequently co-occur → they belong to the same topic → d5 and d6 are semantically related!

    **Topic 1 interpretation:** Gun-violence topic (both terms have high weight)
    """)
    return terms, docs, X_docs, U_docs, S_docs, Vt_docs, cosine_sim, direct_sim_5_6, V_topics, X_topic_space, lsa_sim_5_6, fig5


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## Summary: Key Results

        | Concept | Key Formula | ML Application |
        |---------|-------------|----------------|
        | **Covariance Matrix** | $C = \frac{1}{n}X^TX$ | Data statistics |
        | **PCA** | Eigenvectors of $C$ | Dimensionality reduction |
        | **Variance Explained** | $\lambda_i / \sum \lambda_j$ | Feature selection |
        | **SVD** | $A = U\Sigma V^T$ | Matrix factorization |
        | **Low-Rank Approx** | Keep top $r$ singular values | Compression, denoising |
        | **LSA** | SVD on doc-term matrix | Semantic similarity |

        ### Key Insights:

        1. **PCA finds directions of maximum variance** via covariance eigendecomposition
        2. **SVD generalizes eigendecomposition** to any matrix
        3. **Low-rank approximation** keeps signal, discards noise
        4. **LSA reveals semantic similarity** beyond exact term matching

        ---

        *IME 775 - Mathematical Foundations of Deep Learning*
        """
    )
    return


if __name__ == "__main__":
    app.run()
