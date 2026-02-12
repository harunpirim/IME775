"""
IME 775: Quadratic Forms and Their Optimization
================================================
A marimo notebook exploring quadratic forms, their geometric meaning,
and how eigenvalues determine extrema.

Course: IME 775 - Mathematical Foundations of Deep Learning
Topics: Quadratic Forms, Positive Definiteness, Matrix Norms
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
        # IME 775: Quadratic Forms and Their Optimization

        ## Learning Objectives

        1. Understand quadratic forms and their geometric interpretation
        2. Visualize how eigenvalues determine the shape of quadratic forms
        3. Prove that extrema occur along eigenvector directions
        4. Explore positive definite matrices and their properties

        ---
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    return np, plt, Axes3D, cm


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. What is a Quadratic Form?

        Given a symmetric matrix $A$, the **quadratic form** is:

        $$Q(\vec{x}) = \vec{x}^T A \vec{x} = \sum_{i,j} a_{ij} x_i x_j$$

        For a 2D vector $\vec{x} = [x_0, x_1]^T$ and matrix $A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$:

        $$Q = ax_0^2 + 2bx_0x_1 + cx_1^2$$

        **Key Insight:** Quadratic forms generalize conic sections (circles, ellipses, hyperbolas) to arbitrary dimensions.
        """
    )
    return


@app.cell
def __(mo):
    # Interactive matrix builder for quadratic form
    qf_a = mo.ui.slider(-3, 3, value=2, step=0.25, label="A[0,0] = a")
    qf_b = mo.ui.slider(-3, 3, value=0.5, step=0.25, label="A[0,1] = A[1,0] = b")
    qf_c = mo.ui.slider(-3, 3, value=1, step=0.25, label="A[1,1] = c")

    mo.md(f"""
    ### Build Your Symmetric Matrix

    $$A = \\begin{{bmatrix}} a & b \\\\ b & c \\end{{bmatrix}}$$

    {qf_a}
    {qf_b}
    {qf_c}
    """)
    return qf_a, qf_b, qf_c


@app.cell
def __(np, plt, qf_a, qf_b, qf_c, cm, mo):
    # Build matrix
    A_qf = np.array([[qf_a.value, qf_b.value],
                     [qf_b.value, qf_c.value]])

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A_qf)

    # Create meshgrid for visualization
    x0 = np.linspace(-2, 2, 100)
    x1 = np.linspace(-2, 2, 100)
    X0, X1 = np.meshgrid(x0, x1)

    # Compute quadratic form Q = x^T A x
    Q = A_qf[0,0]*X0**2 + 2*A_qf[0,1]*X0*X1 + A_qf[1,1]*X1**2

    # Determine matrix type
    if eigvals[0] > 0 and eigvals[1] > 0:
        matrix_type = "Positive Definite (bowl shape)"
        color = 'green'
    elif eigvals[0] < 0 and eigvals[1] < 0:
        matrix_type = "Negative Definite (inverted bowl)"
        color = 'red'
    elif eigvals[0] * eigvals[1] < 0:
        matrix_type = "Indefinite (saddle shape)"
        color = 'orange'
    else:
        matrix_type = "Positive/Negative Semidefinite"
        color = 'blue'

    # Create visualization
    fig = plt.figure(figsize=(16, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X0, X1, Q, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    ax1.set_xlabel('$x_0$')
    ax1.set_ylabel('$x_1$')
    ax1.set_zlabel('$Q(x)$')
    ax1.set_title(f'Quadratic Form Surface\n{matrix_type}')

    # Contour plot with eigenvectors
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X0, X1, Q, levels=20, cmap='coolwarm')
    ax2.set_xlabel('$x_0$')
    ax2.set_ylabel('$x_1$')
    ax2.set_title('Contour Plot with Eigenvectors')
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)

    # Draw eigenvectors
    scale = 1.5
    for i, (val, vec) in enumerate(zip(eigvals, eigvecs.T)):
        color_vec = 'blue' if i == 0 else 'red'
        ax2.quiver(0, 0, scale*vec[0], scale*vec[1], angles='xy', scale_units='xy', scale=1,
                   color=color_vec, width=0.03, label=f'$\\vec{{e}}_{i+1}$ (λ={val:.2f})')
        ax2.quiver(0, 0, -scale*vec[0], -scale*vec[1], angles='xy', scale_units='xy', scale=1,
                   color=color_vec, width=0.03, alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    # Info panel
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    info_text = f"""
    Matrix Analysis
    ═══════════════════════════

    A = [{A_qf[0,0]:6.2f}  {A_qf[0,1]:6.2f}]
        [{A_qf[1,0]:6.2f}  {A_qf[1,1]:6.2f}]

    ───────────────────────────

    Eigenvalues:
    λ₁ = {eigvals[0]:.3f}
    λ₂ = {eigvals[1]:.3f}

    Eigenvectors:
    e₁ = [{eigvecs[0,0]:.3f}, {eigvecs[1,0]:.3f}]ᵀ
    e₂ = [{eigvecs[0,1]:.3f}, {eigvecs[1,1]:.3f}]ᵀ

    ───────────────────────────

    Matrix Type: {matrix_type}

    ───────────────────────────

    Quadratic Form:
    Q = {A_qf[0,0]:.2f}x₀² + {2*A_qf[0,1]:.2f}x₀x₁ + {A_qf[1,1]:.2f}x₁²

    For unit vectors:
    • Maximum Q = λ₂ = {eigvals[1]:.3f}
      (along e₂)
    • Minimum Q = λ₁ = {eigvals[0]:.3f}
      (along e₁)
    """

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.gca()
    return A_qf, eigvals, eigvecs, fig, matrix_type


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 2. Extrema of Quadratic Forms on Unit Sphere

        **Key Question:** For unit vectors $\hat{x}$ (where $\|\hat{x}\| = 1$), what values of $Q = \hat{x}^T A \hat{x}$ are possible?

        **Theorem:** Using spectral decomposition $A = S\Lambda S^T$:

        $$Q = \hat{x}^T A \hat{x} = \hat{y}^T \Lambda \hat{y} = \sum_{i=1}^n \lambda_i y_i^2$$

        where $\hat{y} = S^T \hat{x}$ is also a unit vector.

        Since $\sum_i y_i^2 = 1$, $Q$ is a weighted average of eigenvalues with non-negative weights summing to 1.

        **Result:**
        - **Maximum:** $Q_{max} = \lambda_{max}$ achieved when $\hat{x}$ is the eigenvector for largest eigenvalue
        - **Minimum:** $Q_{min} = \lambda_{min}$ achieved when $\hat{x}$ is the eigenvector for smallest eigenvalue
        """
    )
    return


@app.cell
def __(mo):
    # Interactive unit vector explorer
    theta_unit = mo.ui.slider(0, 360, value=45, step=5, label="Angle θ (degrees)")

    mo.md(f"""
    ### Explore Q on the Unit Circle

    Move around the unit circle to see how $Q$ varies:

    {theta_unit}
    """)
    return theta_unit,


@app.cell
def __(np, plt, qf_a, qf_b, qf_c, theta_unit, mo):
    # Rebuild matrix
    A_unit = np.array([[qf_a.value, qf_b.value],
                       [qf_b.value, qf_c.value]])
    evals, evecs = np.linalg.eigh(A_unit)

    # Current unit vector
    theta_rad = np.radians(theta_unit.value)
    x_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Compute Q for this unit vector
    Q_current = x_unit @ A_unit @ x_unit

    # Compute Q for all angles
    thetas = np.linspace(0, 2*np.pi, 360)
    Q_all = []
    for t in thetas:
        x_t = np.array([np.cos(t), np.sin(t)])
        Q_all.append(x_t @ A_unit @ x_t)
    Q_all = np.array(Q_all)

    # Find eigenvector angles
    eig_angles = [np.degrees(np.arctan2(evecs[1,i], evecs[0,i])) % 360 for i in range(2)]

    # Visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Unit circle with current vector
    ax1 = axes2[0]
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax1.add_patch(circle)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Draw eigenvectors
    for i in range(2):
        color_e = 'blue' if i == 0 else 'red'
        label = f'$e_{i+1}$ (λ={evals[i]:.2f})'
        ax1.quiver(0, 0, evecs[0,i], evecs[1,i], angles='xy', scale_units='xy', scale=1,
                   color=color_e, width=0.02, label=label)

    # Draw current vector
    ax1.quiver(0, 0, x_unit[0], x_unit[1], angles='xy', scale_units='xy', scale=1,
               color='green', width=0.03, label=f'$\\hat{{x}}$ (θ={theta_unit.value}°)')
    ax1.scatter([x_unit[0]], [x_unit[1]], color='green', s=100, zorder=5)

    ax1.set_xlabel('$x_0$')
    ax1.set_ylabel('$x_1$')
    ax1.set_title('Unit Circle with Eigenvectors')
    ax1.legend(loc='upper left')

    # Right: Q as function of angle
    ax2 = axes2[1]
    ax2.plot(np.degrees(thetas), Q_all, 'b-', linewidth=2, label='$Q(\\theta)$')
    ax2.axhline(y=evals[0], color='blue', linestyle='--', label=f'λ₁ = {evals[0]:.2f} (min)')
    ax2.axhline(y=evals[1], color='red', linestyle='--', label=f'λ₂ = {evals[1]:.2f} (max)')
    ax2.scatter([theta_unit.value], [Q_current], color='green', s=150, zorder=5,
                label=f'Current: Q = {Q_current:.3f}')

    # Mark eigenvector positions
    for i, ang in enumerate(eig_angles):
        ax2.axvline(x=ang, color='purple', linestyle=':', alpha=0.5)
        ax2.axvline(x=(ang + 180) % 360, color='purple', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Angle θ (degrees)')
    ax2.set_ylabel('$Q = \\hat{x}^T A \\hat{x}$')
    ax2.set_title('Quadratic Form on Unit Circle')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)

    plt.tight_layout()

    mo.md(f"""
    ### Current Values

    **Unit vector:** $\\hat{{x}} = [{x_unit[0]:.3f}, {x_unit[1]:.3f}]^T$

    **Quadratic form:** $Q = \\hat{{x}}^T A \\hat{{x}} = {Q_current:.4f}$

    **Bounds:** $\\lambda_1 = {evals[0]:.3f} \\leq Q \\leq \\lambda_2 = {evals[1]:.3f}$

    **Observation:** $Q$ oscillates between eigenvalues as we traverse the unit circle. Extrema occur exactly at eigenvector directions!
    """)
    return A_unit, evals, evecs, x_unit, Q_current, theta_rad, thetas, Q_all, eig_angles, fig2


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 3. Positive Definite Matrices

        A symmetric matrix $A$ is:
        - **Positive Definite:** $\vec{x}^T A \vec{x} > 0$ for all $\vec{x} \neq 0$
        - **Positive Semidefinite:** $\vec{x}^T A \vec{x} \geq 0$ for all $\vec{x}$
        - **Negative Definite:** $\vec{x}^T A \vec{x} < 0$ for all $\vec{x} \neq 0$
        - **Indefinite:** $\vec{x}^T A \vec{x}$ can be positive or negative

        **Theorem:** $A$ is positive (semi)definite iff all eigenvalues are positive (non-negative).

        ### Why This Matters in ML:
        - **Covariance matrices** are always positive semidefinite
        - **Hessian matrices** at local minima are positive definite
        - **Convex optimization** requires positive definite Hessians
        """
    )
    return


@app.cell
def __(mo):
    # Positive definite explorer
    pd_a = mo.ui.slider(0.1, 5, value=2, step=0.1, label="λ₁ (first eigenvalue)")
    pd_b = mo.ui.slider(0.1, 5, value=1, step=0.1, label="λ₂ (second eigenvalue)")
    pd_theta = mo.ui.slider(0, 90, value=30, step=5, label="Eigenvector rotation (degrees)")

    mo.md(f"""
    ### Build a Positive Definite Matrix

    Control the eigenvalues and eigenvector orientation:

    {pd_a}
    {pd_b}
    {pd_theta}
    """)
    return pd_a, pd_b, pd_theta


@app.cell
def __(np, plt, pd_a, pd_b, pd_theta, cm, mo):
    # Build PD matrix from eigenvalues and rotation
    lambda1 = pd_a.value
    lambda2 = pd_b.value
    rot_angle = np.radians(pd_theta.value)

    # Rotation matrix
    R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                  [np.sin(rot_angle), np.cos(rot_angle)]])

    # Build A = R * Lambda * R^T
    Lambda = np.diag([lambda1, lambda2])
    A_pd = R @ Lambda @ R.T

    # Create contour plot
    x_pd = np.linspace(-3, 3, 200)
    y_pd = np.linspace(-3, 3, 200)
    X_pd, Y_pd = np.meshgrid(x_pd, y_pd)
    Q_pd = A_pd[0,0]*X_pd**2 + 2*A_pd[0,1]*X_pd*Y_pd + A_pd[1,1]*Y_pd**2

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    # Contour plot (level sets are ellipses)
    ax1 = axes3[0]
    levels = [0.5, 1, 2, 3, 4, 5]
    contour = ax1.contour(X_pd, Y_pd, Q_pd, levels=levels, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8, fmt='Q=%.1f')
    ax1.set_xlabel('$x_0$')
    ax1.set_ylabel('$x_1$')
    ax1.set_title('Level Sets of Positive Definite Quadratic Form\n(Ellipses centered at origin)')
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Draw eigenvectors (axes of ellipse)
    evecs_pd = R
    for i in range(2):
        scale = 2
        ax1.quiver(0, 0, scale*evecs_pd[0,i], scale*evecs_pd[1,i],
                   angles='xy', scale_units='xy', scale=1,
                   color='red' if i == 0 else 'blue', width=0.02,
                   label=f'Axis {i+1} (λ={[lambda1, lambda2][i]:.1f})')
    ax1.legend()

    # 3D surface
    ax2 = axes3[1]
    ax2 = fig3.add_subplot(122, projection='3d')
    ax2.plot_surface(X_pd, Y_pd, Q_pd, cmap=cm.viridis, alpha=0.8)
    ax2.set_xlabel('$x_0$')
    ax2.set_ylabel('$x_1$')
    ax2.set_zlabel('$Q$')
    ax2.set_title('Positive Definite = Bowl Shape\n(Global minimum at origin)')

    plt.tight_layout()

    # Compute condition number
    cond_num = max(lambda1, lambda2) / min(lambda1, lambda2)

    mo.md(f"""
    ### Matrix Properties

    **Eigenvalues:** λ₁ = {lambda1:.2f}, λ₂ = {lambda2:.2f}

    **Condition Number:** κ = λ_max/λ_min = {cond_num:.2f}

    **Interpretation:**
    - Level sets are **ellipses** (not hyperbolas!)
    - The quadratic form has a **unique global minimum** at the origin
    - Larger condition number → more elongated ellipse → slower gradient descent
    - **Ideal for optimization:** κ close to 1 (circular level sets)
    """)
    return lambda1, lambda2, rot_angle, R, Lambda, A_pd, Q_pd, fig3, cond_num


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## 4. Matrix Norms

        ### Spectral Norm

        $$\|A\|_2 = \max_{\|\hat{x}\|=1} \|A\hat{x}\| = \sigma_1$$

        The spectral norm is the **largest singular value** (maximum amplification factor).

        ### Frobenius Norm

        $$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\sum_i \sigma_i^2}$$

        The Frobenius norm is the "size" of a matrix (like L2 norm for vectors).

        ### ML Applications:
        - **Spectral norm:** Lipschitz constant of linear layers, spectral normalization
        - **Frobenius norm:** Matrix approximation error, regularization
        """
    )
    return


@app.cell
def __(mo):
    # Matrix norm explorer
    norm_a = mo.ui.slider(-3, 3, value=2, step=0.25, label="A[0,0]")
    norm_b = mo.ui.slider(-3, 3, value=1, step=0.25, label="A[0,1]")
    norm_c = mo.ui.slider(-3, 3, value=0, step=0.25, label="A[1,0]")
    norm_d = mo.ui.slider(-3, 3, value=1.5, step=0.25, label="A[1,1]")

    mo.md(f"""
    ### Explore Matrix Norms

    Build a general (not necessarily symmetric) matrix:

    {norm_a} {norm_b}

    {norm_c} {norm_d}
    """)
    return norm_a, norm_b, norm_c, norm_d


@app.cell
def __(np, plt, norm_a, norm_b, norm_c, norm_d, mo):
    # Build matrix
    A_norm = np.array([[norm_a.value, norm_b.value],
                       [norm_c.value, norm_d.value]])

    # Compute SVD
    U, S, Vt = np.linalg.svd(A_norm)

    # Norms
    spectral_norm = S[0]  # Largest singular value
    frobenius_norm = np.sqrt(np.sum(S**2))
    frobenius_direct = np.sqrt(np.sum(A_norm**2))

    # Visualize unit circle transformation
    theta_circ = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.vstack([np.cos(theta_circ), np.sin(theta_circ)])

    # Transform
    transformed = A_norm @ unit_circle

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Unit circle to ellipse
    ax1 = axes4[0]
    ax1.plot(unit_circle[0], unit_circle[1], 'b-', linewidth=2, label='Unit circle')
    ax1.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed (ellipse)')

    # Show singular vectors
    # Right singular vectors (input)
    for i in range(2):
        ax1.quiver(0, 0, Vt[i,0], Vt[i,1], angles='xy', scale_units='xy', scale=1,
                   color='blue', width=0.02, alpha=0.7)

    # Left singular vectors scaled by singular values (output)
    for i in range(2):
        ax1.quiver(0, 0, S[i]*U[0,i], S[i]*U[1,i], angles='xy', scale_units='xy', scale=1,
                   color='red', width=0.02, alpha=0.7)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    ax1.set_title(f'Unit Circle → Ellipse\nSpectral Norm = {spectral_norm:.3f} (max stretch)')
    ax1.grid(True, alpha=0.3)

    # Right: Singular values bar chart
    ax2 = axes4[1]
    ax2.bar([0, 1], S, color=['red', 'blue'], alpha=0.7)
    ax2.axhline(y=spectral_norm, color='red', linestyle='--', label=f'Spectral: σ₁ = {spectral_norm:.3f}')
    ax2.axhline(y=frobenius_norm, color='green', linestyle='--', label=f'Frobenius: √(σ₁²+σ₂²) = {frobenius_norm:.3f}')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['σ₁', 'σ₂'])
    ax2.set_ylabel('Singular Value')
    ax2.set_title('Singular Values')
    ax2.legend()

    plt.tight_layout()

    mo.md(f"""
    ### Norm Comparison

    | Norm | Formula | Value |
    |------|---------|-------|
    | **Spectral** | $\\|A\\|_2 = \\sigma_1$ | **{spectral_norm:.4f}** |
    | **Frobenius** | $\\|A\\|_F = \\sqrt{{\\sum_i \\sigma_i^2}}$ | **{frobenius_norm:.4f}** |

    **Singular Values:** σ₁ = {S[0]:.3f}, σ₂ = {S[1]:.3f}

    **Geometric Interpretation:**
    - The unit circle maps to an ellipse with semi-axes σ₁ and σ₂
    - Spectral norm = length of longest semi-axis
    - Frobenius norm = √(sum of squared semi-axes)
    """)
    return A_norm, U, S, Vt, spectral_norm, frobenius_norm, fig4


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        ## Summary: Key Takeaways

        | Concept | Key Result | ML Application |
        |---------|------------|----------------|
        | **Quadratic Form** | $Q = \vec{x}^T A \vec{x}$ | Loss surfaces, Taylor expansion |
        | **Extrema** | Max/min at eigenvectors | PCA, optimization |
        | **Positive Definite** | All $\lambda_i > 0$ | Convex losses, valid covariance |
        | **Condition Number** | $\kappa = \lambda_{max}/\lambda_{min}$ | Gradient descent speed |
        | **Spectral Norm** | $\|A\|_2 = \sigma_1$ | Lipschitz constants |
        | **Frobenius Norm** | $\|A\|_F = \sqrt{\sum \sigma_i^2}$ | Approximation error |

        ### Next Steps

        In the next notebook, we'll explore:
        - Principal Component Analysis (PCA)
        - Dimensionality reduction
        - Variance maximization

        ---

        *IME 775 - Mathematical Foundations of Deep Learning*
        """
    )
    return


if __name__ == "__main__":
    app.run()
