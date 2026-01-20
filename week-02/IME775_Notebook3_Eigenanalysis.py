"""
IME 775: Eigenvalues, Eigenvectors, and Spectral Analysis
==========================================================
A marimo notebook exploring eigendecomposition and its applications in ML.

Course: IME 775 - Mathematical Foundations of Deep Learning
Topics: Eigenvalues, Eigenvectors, Diagonalization, PCA
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # IME 775: Eigenvalues, Eigenvectors, and Spectral Analysis

    ## Learning Objectives

    1. Understand eigenvalues and eigenvectors geometrically
    2. Visualize how linear transforms affect eigenvectors
    3. Apply eigendecomposition to symmetric matrices
    4. Connect eigenanalysis to Principal Component Analysis (PCA)

    ---
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Ellipse
    import matplotlib.patches as mpatches
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. What Are Eigenvectors?

    For a square matrix $\mathbf{A}$, an **eigenvector** $\mathbf{v}$ is a special direction that only gets scaled (not rotated) by the transformation:

    $$\mathbf{Av} = \lambda\mathbf{v}$$

    - $\mathbf{v}$: eigenvector (the special direction)
    - $\lambda$: eigenvalue (the scaling factor)

    **Geometric Intuition:** When you apply transformation $\mathbf{A}$ to an eigenvector, it stays on the same line through the origin—it just gets stretched or shrunk by factor $\lambda$.
    """)
    return


@app.cell
def _(mo):
    # Matrix selector for eigenvalue exploration
    matrix_type = mo.ui.dropdown(
        options={
            "Symmetric (Stretch)": "symmetric_stretch",
            "Symmetric (Different scales)": "symmetric_scales",
            "Rotation 45°": "rotation",
            "Shear": "shear",
            "Custom": "custom"
        },
        value="symmetric_stretch",
        label="Select matrix type"
    )

    # Custom matrix inputs
    c_a11 = mo.ui.slider(-3, 3, value=2, step=0.25, label="A[1,1]")
    c_a12 = mo.ui.slider(-3, 3, value=0, step=0.25, label="A[1,2]")
    c_a21 = mo.ui.slider(-3, 3, value=0, step=0.25, label="A[2,1]")
    c_a22 = mo.ui.slider(-3, 3, value=1, step=0.25, label="A[2,2]")

    mo.md(f"""
    ### Explore Eigenvectors Visually

    {matrix_type}

    **Custom matrix (when "Custom" selected):**

    {c_a11} {c_a12}

    {c_a21} {c_a22}
    """)
    return c_a11, c_a12, c_a21, c_a22, matrix_type


@app.cell
def _(ax2, c_a11, c_a12, c_a21, c_a22, matrix_type, mo, np, plt):
    # Define matrices
    def get_example_matrix(mtype):
        if mtype == "symmetric_stretch":
            return np.array([[2, 0], [0, 1]]), "Diagonal (stretch)"
        elif mtype == "symmetric_scales":
            return np.array([[3, 1], [1, 2]]), "Symmetric"
        elif mtype == "rotation":
            theta = np.pi/4
            return np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]]), "Rotation"
        elif mtype == "shear":
            return np.array([[1, 1], [0, 1]]), "Shear"
        else:  # custom
            return np.array([[c_a11.value, c_a12.value],
                           [c_a21.value, c_a22.value]]), "Custom"

    A_eig, matrix_name = get_example_matrix(matrix_type.value)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A_eig)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate points on unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta_circle)
    circle_y = np.sin(theta_circle)
    circle_points = np.vstack([circle_x, circle_y])

    # Transform the circle
    transformed = A_eig @ circle_points

    # Left plot: Original space with eigenvectors
    _ax1 = axes[0]
    _ax1.set_xlim(-4, 4)
    _ax1.set_ylim(-4, 4)
    _ax1.axhline(y=0, color='k', linewidth=0.5)
    _ax1.axvline(x=0, color='k', linewidth=0.5)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_aspect('equal')
    _ax1.set_title('Original Space with Eigenvectors', fontsize=12)

    # Draw unit circle
    _ax1.plot(circle_x, circle_y, 'b-', alpha=0.5, linewidth=2, label='Unit circle')

    # Draw eigenvectors (if real)
    _colors = ['red', 'green']
    for _i in range(2):
        if np.isreal(eigenvalues[_i]):
            _ev = eigenvectors[:, _i].real
            # Normalize for display
            _ev_norm = _ev / np.linalg.norm(_ev) * 1.5
            _ax1.quiver(0, 0, _ev_norm[0], _ev_norm[1], angles='xy', scale_units='xy', scale=1,
                      color=_colors[_i], width=0.03, label=f'v{_i+1} (λ={eigenvalues[_i].real:.2f})')
            # Draw in both directions
            _ax1.quiver(0, 0, -_ev_norm[0], -_ev_norm[1], angles='xy', scale_units='xy', scale=1,
                      color=_colors[_i], width=0.03, alpha=0.3)

    _ax1.legend(loc='upper left')
    _ax1.set_xlabel('x₁')
    _ax1.set_ylabel('x₂')

    # Right plot: Transformed space
    _ax2 = axes[1]
    _ax2.set_xlim(-4, 4)
    _ax2.set_ylim(-4, 4)
    _ax2.axhline(y=0, color='k', linewidth=0.5)
    _ax2.axvline(x=0, color='k', linewidth=0.5)
    _ax2.grid(True, alpha=0.3)
    _ax2.set_aspect('equal')
    _ax2.set_title('Transformed Space (Av)', fontsize=12)

    # Draw transformed circle (becomes ellipse)
    _ax2.plot(transformed[0, :], transformed[1, :], 'purple', alpha=0.7, linewidth=2, label='Transformed circle')

    # Draw transformed eigenvectors
    for _i in range(2):
        if np.isreal(eigenvalues[_i]):
            _ev = eigenvectors[:, _i].real
            _ev_norm = _ev / np.linalg.norm(_ev) * 1.5
            _Aev = A_eig @ _ev_norm
            _ax2.quiver(0, 0, _Aev[0], _Aev[1], angles='xy', scale_units='xy', scale=1,
                      color=_colors[_i], width=0.03, label=f'Av{_i+1} = {eigenvalues[_i].real:.2f}·v{_i+1}')
            _ax2.quiver(0, 0, -_Aev[0], -_Aev[1], angles='xy', scale_units='xy', scale=1,
                      color=_colors[_i], width=0.03, alpha=0.3)

    _ax2.legend(loc='upper left')
    _ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')

    plt.tight_layout()

    # Analysis text
    eig_real = all(np.isreal(eigenvalues))

    if eig_real:
        eig_analysis = f"""
    **Eigenvalues:** λ₁ = {eigenvalues[0].real:.3f}, λ₂ = {eigenvalues[1].real:.3f}

    **Eigenvectors:**
    - v₁ = [{eigenvectors[0,0].real:.3f}, {eigenvectors[1,0].real:.3f}]ᵀ
    - v₂ = [{eigenvectors[0,1].real:.3f}, {eigenvectors[1,1].real:.3f}]ᵀ

    **Interpretation:** Each eigenvector gets scaled by its eigenvalue. The unit circle transforms into an ellipse whose axes align with the eigenvectors.
        """
    else:
        eig_analysis = f"""
    **Eigenvalues:** λ₁ = {eigenvalues[0]:.3f}, λ₂ = {eigenvalues[1]:.3f}

    **Note:** Complex eigenvalues! This matrix involves rotation.
    For rotation by angle θ: λ = cos(θ) ± i·sin(θ)

    The imaginary part encodes the rotation angle.
        """

    mo.md(f"""
    ### Matrix: {matrix_name}

    ```
    A = [{A_eig[0,0]:6.2f}  {A_eig[0,1]:6.2f}]
        [{A_eig[1,0]:6.2f}  {A_eig[1,1]:6.2f}]
    ```

    {eig_analysis}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. The Spectral Theorem for Symmetric Matrices

    For **symmetric matrices** ($\mathbf{A} = \mathbf{A}^T$), we have a beautiful result:

    **Spectral Theorem:**
    1. All eigenvalues are **real**
    2. Eigenvectors are **orthogonal** (perpendicular)
    3. $\mathbf{A}$ can be decomposed as:

    $$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i\mathbf{q}_i^T$$

    where $\mathbf{Q}$ is orthogonal (columns are eigenvectors) and $\boldsymbol{\Lambda}$ is diagonal (eigenvalues).

    **Why This Matters for ML:**
    - Covariance matrices are symmetric → can always be diagonalized
    - PCA uses this decomposition directly
    """)
    return


@app.cell
def _(mo):
    # Symmetric matrix builder
    sym_a = mo.ui.slider(-3, 3, value=3, step=0.25, label="a (diagonal)")
    sym_b = mo.ui.slider(-3, 3, value=1, step=0.25, label="b (off-diagonal)")
    sym_d = mo.ui.slider(-3, 3, value=2, step=0.25, label="d (diagonal)")

    mo.md(f"""
    ### Build a Symmetric Matrix

    The matrix will be:
    ```
    A = [a  b]
        [b  d]
    ```

    {sym_a}
    {sym_b}
    {sym_d}
    """)
    return sym_a, sym_b, sym_d


@app.cell
def _(mo, np, plt, sym_a, sym_b, sym_d):
    # Build symmetric matrix
    A_sym = np.array([[sym_a.value, sym_b.value],
                      [sym_b.value, sym_d.value]])

    # Eigendecomposition
    eigvals_sym, Q_sym = np.linalg.eigh(A_sym)  # eigh for symmetric matrices

    # Verify orthogonality
    orthogonality_check = Q_sym.T @ Q_sym

    # Reconstruct matrix
    Lambda_sym = np.diag(eigvals_sym)
    A_reconstructed = Q_sym @ Lambda_sym @ Q_sym.T

    # Visualization
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Original matrix as heatmap
    _ax1 = axes2[0]
    _im1 = _ax1.imshow(A_sym, cmap='RdBu_r', vmin=-4, vmax=4)
    _ax1.set_title('Original Matrix A')
    for _i in range(2):
        for _j in range(2):
            _ax1.text(_j, _i, f'{A_sym[_i,_j]:.2f}', ha='center', va='center', fontsize=14)
    plt.colorbar(_im1, ax=_ax1, fraction=0.046)

    # Middle: Q and Lambda
    _ax2 = axes2[1]
    _ax2.axis('off')

    decomp_text = f"""
    Spectral Decomposition: A = QΛQᵀ

    ═══════════════════════════════════

    Eigenvalues (Λ):
    λ₁ = {eigvals_sym[0]:.3f}
    λ₂ = {eigvals_sym[1]:.3f}

    ───────────────────────────────────

    Eigenvector matrix (Q):
    q₁ = [{Q_sym[0,0]:.3f}, {Q_sym[1,0]:.3f}]ᵀ
    q₂ = [{Q_sym[0,1]:.3f}, {Q_sym[1,1]:.3f}]ᵀ

    ───────────────────────────────────

    Orthogonality check (QᵀQ):
    [{orthogonality_check[0,0]:.4f}  {orthogonality_check[0,1]:.4f}]
    [{orthogonality_check[1,0]:.4f}  {orthogonality_check[1,1]:.4f}]

    (Should be identity matrix ✓)

    ───────────────────────────────────

    Reconstruction error:
    ‖A - QΛQᵀ‖ = {np.linalg.norm(A_sym - A_reconstructed):.2e}
    """

    _ax2.text(0.1, 0.95, decomp_text, transform=_ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Right: Eigenvectors visualization
    _ax3 = axes2[2]
    _ax3.set_xlim(-2, 2)
    _ax3.set_ylim(-2, 2)
    _ax3.axhline(y=0, color='k', linewidth=0.5)
    _ax3.axvline(x=0, color='k', linewidth=0.5)
    _ax3.grid(True, alpha=0.3)
    _ax3.set_aspect('equal')
    _ax3.set_title('Orthogonal Eigenvectors')

    # Draw eigenvectors
    _ax3.quiver(0, 0, Q_sym[0,0], Q_sym[1,0], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.03, label=f'q₁ (λ₁={eigvals_sym[0]:.2f})')
    _ax3.quiver(0, 0, Q_sym[0,1], Q_sym[1,1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.03, label=f'q₂ (λ₂={eigvals_sym[1]:.2f})')

    # Draw unit circle for reference
    _theta_c = np.linspace(0, 2*np.pi, 100)
    _ax3.plot(np.cos(_theta_c), np.sin(_theta_c), 'gray', alpha=0.3, linestyle='--')

    _ax3.legend(loc='upper right')
    _ax3.set_xlabel('x₁')
    _ax3.set_ylabel('x₂')

    plt.tight_layout()

    # Check angle between eigenvectors
    dot_eigs = np.dot(Q_sym[:,0], Q_sym[:,1])
    angle_eigs = np.degrees(np.arccos(np.clip(np.abs(dot_eigs), 0, 1)))

    mo.md(f"""
    ### Verification

    **Angle between eigenvectors:** {90 - angle_eigs:.2f}° from perpendicular (should be ~0°)

    **Dot product of eigenvectors:** q₁ · q₂ = {dot_eigs:.6f} (should be ~0)

    **Key Result:** For symmetric matrices, eigenvectors are always orthogonal!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Matrix Diagonalization: Powers Made Easy

    When $\mathbf{A} = \mathbf{S}\boldsymbol{\Lambda}\mathbf{S}^{-1}$, computing matrix powers becomes trivial:

    $$\mathbf{A}^k = \mathbf{S}\boldsymbol{\Lambda}^k\mathbf{S}^{-1}$$

    Since $\boldsymbol{\Lambda}$ is diagonal:
    $$\boldsymbol{\Lambda}^k = \text{diag}(\lambda_1^k, \lambda_2^k, \ldots, \lambda_n^k)$$

    **Applications in ML:**
    - Markov chains (long-term behavior)
    - Recurrent neural networks (gradient analysis)
    - Graph neural networks (message passing)
    """)
    return


@app.cell
def _(mo):
    # Matrix power exploration
    power_k = mo.ui.slider(1, 20, value=1, step=1, label="Power k")
    power_l1 = mo.ui.slider(0.1, 2, value=0.9, step=0.05, label="λ₁ (eigenvalue 1)")
    power_l2 = mo.ui.slider(0.1, 2, value=0.5, step=0.05, label="λ₂ (eigenvalue 2)")

    mo.md(f"""
    ### Explore Matrix Powers via Diagonalization

    {power_k}
    {power_l1}
    {power_l2}
    """)
    return power_k, power_l1, power_l2


@app.cell
def _(np, plt, power_k, power_l1, power_l2):
    # Create a matrix with specified eigenvalues
    # Use a random orthogonal matrix for eigenvectors
    np.random.seed(42)
    Q_power = np.array([[0.6, -0.8], [0.8, 0.6]])  # Orthogonal
    L1, L2 = power_l1.value, power_l2.value
    Lambda_power = np.diag([L1, L2])

    A_power = Q_power @ Lambda_power @ Q_power.T

    k = power_k.value

    # Compute A^k directly and via diagonalization
    A_k_direct = np.linalg.matrix_power(A_power, k)
    Lambda_k = np.diag([L1**k, L2**k])
    A_k_diag = Q_power @ Lambda_k @ Q_power.T

    # Plot eigenvalue decay/growth
    k_range = np.arange(1, 21)
    l1_powers = L1 ** k_range
    l2_powers = L2 ** k_range

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Eigenvalue powers over k
    _ax1 = axes3[0]
    _ax1.semilogy(k_range, l1_powers, 'ro-', label=f'λ₁^k = {L1:.2f}^k', linewidth=2)
    _ax1.semilogy(k_range, l2_powers, 'bs-', label=f'λ₂^k = {L2:.2f}^k', linewidth=2)
    _ax1.axvline(x=k, color='green', linestyle='--', linewidth=2, label=f'Current k={k}')
    _ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    _ax1.set_xlabel('Power k')
    _ax1.set_ylabel('Eigenvalue^k (log scale)')
    _ax1.set_title('Eigenvalue Powers Over Time')
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)
    _ax1.set_xlim(1, 20)

    # Right: Matrix norms
    _ax2 = axes3[1]
    _ax2.axis('off')

    # Determine behavior
    if L1 > 1 or L2 > 1:
        behavior = "UNSTABLE (eigenvalue > 1 causes growth)"
        _color = 'lightcoral'
    elif L1 < 1 and L2 < 1:
        behavior = "STABLE (all eigenvalues < 1 cause decay)"
        _color = 'lightgreen'
    else:
        behavior = "MARGINAL"
        _color = 'lightyellow'

    power_text = f"""
    Matrix Power Analysis: A^{k}
    ═════════════════════════════════════════

    Original Matrix A:
    [{A_power[0,0]:7.3f}  {A_power[0,1]:7.3f}]
    [{A_power[1,0]:7.3f}  {A_power[1,1]:7.3f}]

    Eigenvalues: λ₁ = {L1:.3f}, λ₂ = {L2:.3f}

    ─────────────────────────────────────────

    After k = {k} iterations:

    λ₁^{k} = {L1**k:.6f}
    λ₂^{k} = {L2**k:.6f}

    A^{k} via diagonalization:
    [{A_k_diag[0,0]:10.5f}  {A_k_diag[0,1]:10.5f}]
    [{A_k_diag[1,0]:10.5f}  {A_k_diag[1,1]:10.5f}]

    ─────────────────────────────────────────

    System behavior: {behavior}

    Dominant eigenvalue: λ = {max(L1, L2):.3f}
    Spectral radius: ρ(A) = {max(L1, L2):.3f}

    ─────────────────────────────────────────

    ML Insight: In recurrent networks, if the
    spectral radius > 1, gradients explode.
    If spectral radius < 1, gradients vanish.
    """

    _ax2.text(0.05, 0.95, power_text, transform=_ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=_color, alpha=0.7))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Principal Component Analysis (PCA)

    PCA uses eigendecomposition to find the directions of maximum variance in data:

    1. Center data: $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{x}}$
    2. Compute covariance: $\mathbf{C} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$
    3. Find eigenvectors of $\mathbf{C}$ (principal components)
    4. Project data onto top $k$ eigenvectors

    The eigenvectors point in the directions of maximum variance. The eigenvalues tell us how much variance each direction captures.
    """)
    return


@app.cell
def _(mo):
    # PCA demo parameters
    pca_n_points = mo.ui.slider(50, 200, value=100, step=10, label="Number of points")
    pca_spread_x = mo.ui.slider(0.5, 3, value=2, step=0.1, label="Spread along PC1")
    pca_spread_y = mo.ui.slider(0.1, 1.5, value=0.5, step=0.1, label="Spread along PC2")
    pca_angle = mo.ui.slider(0, 90, value=30, step=5, label="Rotation angle (degrees)")
    pca_k = mo.ui.slider(1, 2, value=2, step=1, label="Components to keep (k)")

    mo.md(f"""
    ### Interactive PCA Demo

    Generate 2D data with controlled spread and rotation:

    {pca_n_points}
    {pca_spread_x}
    {pca_spread_y}
    {pca_angle}
    {pca_k}
    """)
    return pca_angle, pca_k, pca_n_points, pca_spread_x, pca_spread_y


@app.cell
def _(mo, np, pca_angle, pca_k, pca_n_points, pca_spread_x, pca_spread_y, plt):
    np.random.seed(42)

    # Generate data
    n_pts = pca_n_points.value
    spread_x = pca_spread_x.value
    spread_y = pca_spread_y.value
    angle = np.radians(pca_angle.value)
    k_components = pca_k.value

    # Create elliptical data
    X_orig = np.column_stack([
        spread_x * np.random.randn(n_pts),
        spread_y * np.random.randn(n_pts)
    ])

    # Rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    # Rotate data
    X_data = (R @ X_orig.T).T

    # Add some offset (then center)
    X_data += np.array([2, 1])

    # PCA
    # 1. Center data
    X_mean = np.mean(X_data, axis=0)
    X_centered = X_data - X_mean

    # 2. Covariance matrix
    Cov = np.cov(X_centered.T)

    # 3. Eigendecomposition
    pca_eigenvalues, pca_eigenvectors = np.linalg.eigh(Cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(pca_eigenvalues)[::-1]
    pca_eigenvalues = pca_eigenvalues[idx]
    pca_eigenvectors = pca_eigenvectors[:, idx]

    # 4. Project onto k components
    W = pca_eigenvectors[:, :k_components]
    X_projected = X_centered @ W

    # Reconstruct
    X_reconstructed = X_projected @ W.T + X_mean

    # Variance explained
    total_var = np.sum(pca_eigenvalues)
    var_explained = pca_eigenvalues / total_var * 100

    # Visualization
    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Original data with principal components
    _ax1 = axes4[0]
    _ax1.scatter(X_data[:, 0], X_data[:, 1], alpha=0.5, s=30, c='blue', label='Data')
    _ax1.scatter(X_mean[0], X_mean[1], s=200, c='red', marker='x', linewidths=3, label='Mean')

    # Draw principal components (scaled by eigenvalues)
    _scale = 2
    for _i in range(2):
        _ev = pca_eigenvectors[:, _i] * np.sqrt(pca_eigenvalues[_i]) * _scale
        _color = 'green' if _i == 0 else 'orange'
        _ax1.quiver(X_mean[0], X_mean[1], _ev[0], _ev[1], angles='xy', scale_units='xy', scale=1,
                  color=_color, width=0.03, label=f'PC{_i+1} ({var_explained[_i]:.1f}%)')
        _ax1.quiver(X_mean[0], X_mean[1], -_ev[0], -_ev[1], angles='xy', scale_units='xy', scale=1,
                  color=_color, width=0.03, alpha=0.5)

    _ax1.set_xlabel('x₁')
    _ax1.set_ylabel('x₂')
    _ax1.set_title('Original Data + Principal Components')
    _ax1.legend(loc='upper left')
    _ax1.set_aspect('equal')
    _ax1.grid(True, alpha=0.3)

    # Middle: Projected data
    _ax2 = axes4[1]
    if k_components == 2:
        _ax2.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.5, s=30, c='purple')
        _ax2.set_xlabel('PC1')
        _ax2.set_ylabel('PC2')
    else:
        _ax2.scatter(X_projected[:, 0], np.zeros(n_pts), alpha=0.5, s=30, c='purple')
        _ax2.set_xlabel('PC1')
        _ax2.set_ylabel('(compressed)')
        _ax2.set_ylim(-1, 1)

    _ax2.axhline(y=0, color='k', linewidth=0.5)
    _ax2.axvline(x=0, color='k', linewidth=0.5)
    _ax2.set_title(f'Projected Data (k={k_components})')
    _ax2.grid(True, alpha=0.3)

    # Right: Reconstruction
    _ax3 = axes4[2]
    _ax3.scatter(X_data[:, 0], X_data[:, 1], alpha=0.3, s=30, c='blue', label='Original')
    _ax3.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, s=30, c='red', label='Reconstructed')

    # Draw lines connecting original to reconstructed
    for _i in range(0, n_pts, 5):
        _ax3.plot([X_data[_i, 0], X_reconstructed[_i, 0]],
                [X_data[_i, 1], X_reconstructed[_i, 1]], 'gray', alpha=0.3)

    reconstruction_error = np.mean(np.sum((X_data - X_reconstructed)**2, axis=1))

    _ax3.set_xlabel('x₁')
    _ax3.set_ylabel('x₂')
    _ax3.set_title(f'Reconstruction (Error: {reconstruction_error:.3f})')
    _ax3.legend()
    _ax3.set_aspect('equal')
    _ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md(f"""
    ### PCA Analysis Results

    **Covariance Matrix:**
    ```
    C = [{Cov[0,0]:7.3f}  {Cov[0,1]:7.3f}]
        [{Cov[1,0]:7.3f}  {Cov[1,1]:7.3f}]
    ```

    **Principal Components:**

    | PC | Eigenvalue | Variance Explained | Cumulative |
    |----|------------|-------------------|------------|
    | PC1 | {pca_eigenvalues[0]:.3f} | {var_explained[0]:.1f}% | {var_explained[0]:.1f}% |
    | PC2 | {pca_eigenvalues[1]:.3f} | {var_explained[1]:.1f}% | {var_explained[0]+var_explained[1]:.1f}% |

    **Using k={k_components} components:**
    - Variance retained: **{np.sum(var_explained[:k_components]):.1f}%**
    - Reconstruction MSE: **{reconstruction_error:.4f}**

    **Insight:** PC1 (green arrow) points in the direction of maximum variance!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary: Eigenanalysis in Machine Learning

    | Concept | Formula | ML Application |
    |---------|---------|----------------|
    | **Eigenvector** | $\mathbf{Av} = \lambda\mathbf{v}$ | Directions preserved by transforms |
    | **Eigenvalue** | $\lambda$ | Scaling along eigenvector direction |
    | **Spectral Theorem** | $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ | Decomposition of symmetric matrices |
    | **Matrix Power** | $\mathbf{A}^k = \mathbf{S}\boldsymbol{\Lambda}^k\mathbf{S}^{-1}$ | RNN stability, Markov chains |
    | **PCA** | Project onto top eigenvectors | Dimensionality reduction |
    | **Spectral Radius** | $\rho(\mathbf{A}) = \max_i|\lambda_i|$ | Gradient flow analysis |

    ### Key Insights for Deep Learning:

    1. **Weight Initialization:** Eigenvalue analysis helps design stable initializations
    2. **Optimization:** Condition number (ratio of eigenvalues) affects convergence
    3. **Feature Learning:** PCA reveals intrinsic dimensionality of data
    4. **Graph Networks:** Eigenvalues of graph Laplacian capture connectivity

    ---

    *IME 775 - Mathematical Foundations of Deep Learning*
    """)
    return


if __name__ == "__main__":
    app.run()
