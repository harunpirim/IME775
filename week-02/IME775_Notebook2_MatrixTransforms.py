"""
IME 775: Matrix Operations and Linear Transforms
=================================================
A marimo notebook exploring matrices and their role in machine learning.

Course: IME 775 - Mathematical Foundations of Deep Learning
Topics: Matrices, Linear Transforms, Matrix Multiplication, Systems of Equations
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
    # IME 775: Matrix Operations and Linear Transforms

    ## Learning Objectives

    1. Understand matrices as data structures and linear transforms
    2. Visualize how matrix multiplication transforms vectors
    3. Explore the geometry of linear transformations
    4. Solve linear systems and understand invertibility

    ---
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    return Polygon, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Matrices: From Data to Transformations

    A **matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ serves two fundamental purposes:

    1. **Data Storage**: Store $m$ samples with $n$ features each
    2. **Linear Transform**: Map vectors from $\mathbb{R}^n$ to $\mathbb{R}^m$

    $$\mathbf{A} = \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \cdots & a_{mn}
    \end{bmatrix}$$
    """)
    return


@app.cell
def _(mo):
    # Interactive 2x2 matrix builder
    mo.md("""
    ### Build Your 2×2 Transformation Matrix
    """)
    return


@app.cell
def _(mo):
    a11 = mo.ui.slider(-3, 3, value=1, step=0.25, label="A[1,1]")
    a12 = mo.ui.slider(-3, 3, value=0, step=0.25, label="A[1,2]")
    a21 = mo.ui.slider(-3, 3, value=0, step=0.25, label="A[2,1]")
    a22 = mo.ui.slider(-3, 3, value=1, step=0.25, label="A[2,2]")

    mo.md(f"""
    **Matrix A:**

    Row 1: {a11} {a12}

    Row 2: {a21} {a22}
    """)
    return a11, a12, a21, a22


@app.cell
def _(Polygon, a11, a12, a21, a22, np, plt):
    # Create transformation matrix
    A = np.array([[a11.value, a12.value],
                  [a21.value, a22.value]])

    # Create original shape (unit square)
    original_points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    # Transform the square
    transformed_points = (A @ original_points.T).T

    # Also show basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    Ae1 = A @ e1
    Ae2 = A @ e2

    # Calculate determinant
    det_A = np.linalg.det(A)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left plot: Original
    _ax1 = axes[0]
    _ax1.set_xlim(-3, 3)
    _ax1.set_ylim(-3, 3)
    _ax1.axhline(y=0, color='k', linewidth=0.5)
    _ax1.axvline(x=0, color='k', linewidth=0.5)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_aspect('equal')
    _ax1.set_title('Original Space')

    # Draw original square
    _square = Polygon(original_points, fill=True, alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=2)
    _ax1.add_patch(_square)

    # Draw basis vectors
    _ax1.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.03, label='e₁ = [1, 0]')
    _ax1.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1,
               color='green', width=0.03, label='e₂ = [0, 1]')
    _ax1.legend(loc='upper left')
    _ax1.set_xlabel('x')
    _ax1.set_ylabel('y')

    # Middle plot: Transformed
    _ax2 = axes[1]
    _ax2.set_xlim(-4, 4)
    _ax2.set_ylim(-4, 4)
    _ax2.axhline(y=0, color='k', linewidth=0.5)
    _ax2.axvline(x=0, color='k', linewidth=0.5)
    _ax2.grid(True, alpha=0.3)
    _ax2.set_aspect('equal')
    _ax2.set_title('Transformed Space: A·x')

    # Draw transformed shape
    if not np.isclose(det_A, 0):
        _transformed_square = Polygon(transformed_points, fill=True, alpha=0.3,
                                     facecolor='purple', edgecolor='purple', linewidth=2)
        _ax2.add_patch(_transformed_square)
    else:
        # Collapsed to line
        _ax2.plot(transformed_points[:, 0], transformed_points[:, 1], 'purple', linewidth=3)

    # Draw transformed basis vectors
    _ax2.quiver(0, 0, Ae1[0], Ae1[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.03, label=f'Ae₁ = [{Ae1[0]:.2f}, {Ae1[1]:.2f}]')
    _ax2.quiver(0, 0, Ae2[0], Ae2[1], angles='xy', scale_units='xy', scale=1,
               color='green', width=0.03, label=f'Ae₂ = [{Ae2[0]:.2f}, {Ae2[1]:.2f}]')
    _ax2.legend(loc='upper left')
    _ax2.set_xlabel('x')
    _ax2.set_ylabel('y')

    # Right plot: Matrix info
    ax3 = axes[2]
    ax3.axis('off')

    transform_type = []
    if np.allclose(A, np.eye(2)):
        transform_type.append("Identity (no change)")
    if np.allclose(A, A.T) and not np.allclose(A, np.eye(2)):
        transform_type.append("Symmetric")
    if np.allclose(A @ A.T, np.eye(2) * (det_A**2)) and np.abs(det_A) > 0.01:
        if np.isclose(det_A, 1):
            transform_type.append("Rotation")
        elif np.isclose(det_A, -1):
            transform_type.append("Reflection")
    if np.isclose(A[0,1], 0) and np.isclose(A[1,0], 0):
        transform_type.append("Scaling")
    if (np.isclose(A[0,0], 1) and np.isclose(A[1,1], 1) and
        (not np.isclose(A[0,1], 0) or not np.isclose(A[1,0], 0))):
        transform_type.append("Shear")
    if np.isclose(det_A, 0):
        transform_type.append("Singular (collapses space)")

    transform_str = ", ".join(transform_type) if transform_type else "General linear transform"

    info_text = f"""
    Matrix Analysis
    ═══════════════════════════════

    A = [{A[0,0]:6.2f}  {A[0,1]:6.2f}]
        [{A[1,0]:6.2f}  {A[1,1]:6.2f}]

    ───────────────────────────────

    Determinant: det(A) = {det_A:.3f}

    Area scaling: |det(A)| = {abs(det_A):.3f}
    (Unit square area → {abs(det_A):.3f})

    ───────────────────────────────

    Invertible: {'Yes ✓' if abs(det_A) > 0.001 else 'No ✗ (singular)'}

    Transform type: {transform_str}

    ───────────────────────────────

    Basis vector transformations:
    • e₁ = [1, 0] → [{Ae1[0]:.2f}, {Ae1[1]:.2f}]
    • e₂ = [0, 1] → [{Ae2[0]:.2f}, {Ae2[1]:.2f}]

    Key insight: Columns of A are the
    transformed basis vectors!
    """

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Common Transformation Matrices

    Try these matrices by adjusting the sliders above:

    | Transform | Matrix | Effect |
    |-----------|--------|--------|
    | **Identity** | $\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}$ | No change |
    | **Scale** | $\begin{bmatrix}2 & 0\\0 & 0.5\end{bmatrix}$ | Stretch x, compress y |
    | **Rotation 45°** | $\begin{bmatrix}0.71 & -0.71\\0.71 & 0.71\end{bmatrix}$ | Rotate counter-clockwise |
    | **Shear** | $\begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}$ | Slant horizontally |
    | **Reflection** | $\begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}$ | Mirror over x-axis |
    | **Singular** | $\begin{bmatrix}1 & 2\\0.5 & 1\end{bmatrix}$ | Collapses to line |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Matrix Multiplication: Composing Transforms

    When we multiply matrices $\mathbf{C} = \mathbf{AB}$, we compose transformations:
    - First apply $\mathbf{B}$
    - Then apply $\mathbf{A}$

    $$\mathbf{C}\mathbf{x} = \mathbf{A}(\mathbf{B}\mathbf{x})$$

    **Critical:** Matrix multiplication is NOT commutative: $\mathbf{AB} \neq \mathbf{BA}$

    Let's explore this!
    """)
    return


@app.cell
def _(mo):
    # Preset transformation selector
    transform_B = mo.ui.dropdown(
        options={
            "Identity": "identity",
            "Scale 2x": "scale",
            "Rotate 45°": "rotate45",
            "Rotate 90°": "rotate90",
            "Shear": "shear",
            "Reflect Y": "reflect"
        },
        value="rotate45",
        label="First Transform (B)"
    )

    transform_A = mo.ui.dropdown(
        options={
            "Identity": "identity",
            "Scale 2x": "scale",
            "Rotate 45°": "rotate45",
            "Rotate 90°": "rotate90",
            "Shear": "shear",
            "Reflect Y": "reflect"
        },
        value="scale",
        label="Second Transform (A)"
    )

    mo.md(f"""
    ### Compose Two Transformations

    {transform_B}

    {transform_A}
    """)
    return transform_A, transform_B


@app.cell
def _(Polygon, mo, np, plt, transform_A, transform_B):
    def get_matrix(name):
        if name == "identity":
            return np.eye(2), "I"
        elif name == "scale":
            return np.array([[2, 0], [0, 2]]), "S"
        elif name == "rotate45":
            theta = np.pi/4
            return np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]]), "R₄₅"
        elif name == "rotate90":
            return np.array([[0, -1], [1, 0]]), "R₉₀"
        elif name == "shear":
            return np.array([[1, 0.5], [0, 1]]), "H"
        elif name == "reflect":
            return np.array([[1, 0], [0, -1]]), "F"
        return np.eye(2), "I"

    B, B_name = get_matrix(transform_B.value)
    A_mat, A_name = get_matrix(transform_A.value)

    # Compose: AB and BA
    AB = A_mat @ B
    BA = B @ A_mat

    # Original shape
    orig = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Transformations
    after_B = (B @ orig.T).T
    after_AB = (AB @ orig.T).T
    after_BA = (BA @ orig.T).T

    # Visualization
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

    def plot_shape(ax, points, title, color='blue'):
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        poly = Polygon(points, fill=True, alpha=0.4, facecolor=color, edgecolor=color, linewidth=2)
        ax.add_patch(poly)

    # Top row: AB composition
    plot_shape(axes2[0, 0], orig, f'Original', 'blue')
    plot_shape(axes2[0, 1], after_B, f'After {B_name} (first)', 'orange')
    plot_shape(axes2[0, 2], after_AB, f'After {A_name}·{B_name} (AB)', 'red')

    # Bottom row: BA composition
    after_A = (A_mat @ orig.T).T
    plot_shape(axes2[1, 0], orig, f'Original', 'blue')
    plot_shape(axes2[1, 1], after_A, f'After {A_name} (first)', 'green')
    plot_shape(axes2[1, 2], after_BA, f'After {B_name}·{A_name} (BA)', 'purple')

    plt.tight_layout()

    # Check if commutative
    is_commutative = np.allclose(AB, BA)

    mo.md(f"""
    ### Composition Results

    **Order 1: {A_name}·{B_name}** (first B, then A)
    ```
    AB = {np.array2string(AB, precision=2)}
    ```

    **Order 2: {B_name}·{A_name}** (first A, then B)
    ```
    BA = {np.array2string(BA, precision=2)}
    ```

    **Are they equal?** {'Yes ✓ (commutative for these transforms)' if is_commutative else 'No ✗ (AB ≠ BA)'}

    **Key Insight:** The order of matrix multiplication matters! In neural networks,
    applying layers in different orders produces different results.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Linear Systems: Ax = b

    A fundamental problem in ML is solving linear systems:

    $$\mathbf{Ax} = \mathbf{b}$$

    **Interpretations:**
    - Find input $\mathbf{x}$ that produces output $\mathbf{b}$ under transform $\mathbf{A}$
    - Find weights $\mathbf{x}$ that fit data $\mathbf{b}$

    **Solution (when A is invertible):** $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$
    """)
    return


@app.cell
def _(mo):
    # Linear system explorer
    sys_a11 = mo.ui.slider(-3, 3, value=2, step=0.5, label="A[1,1]")
    sys_a12 = mo.ui.slider(-3, 3, value=1, step=0.5, label="A[1,2]")
    sys_a21 = mo.ui.slider(-3, 3, value=1, step=0.5, label="A[2,1]")
    sys_a22 = mo.ui.slider(-3, 3, value=3, step=0.5, label="A[2,2]")
    sys_b1 = mo.ui.slider(-5, 5, value=3, step=0.5, label="b₁")
    sys_b2 = mo.ui.slider(-5, 5, value=4, step=0.5, label="b₂")

    mo.md(f"""
    ### Solve a Linear System

    **Matrix A:** {sys_a11} {sys_a12} | {sys_a21} {sys_a22}

    **Vector b:** {sys_b1} {sys_b2}
    """)
    return sys_a11, sys_a12, sys_a21, sys_a22, sys_b1, sys_b2


@app.cell
def _(np, plt, sys_a11, sys_a12, sys_a21, sys_a22, sys_b1, sys_b2):
    # Build system
    A_sys = np.array([[sys_a11.value, sys_a12.value],
                      [sys_a21.value, sys_a22.value]])
    b_sys = np.array([sys_b1.value, sys_b2.value])

    det_sys = np.linalg.det(A_sys)

    # Solve if possible
    if np.abs(det_sys) > 1e-6:
        x_sol = np.linalg.solve(A_sys, b_sys)
        has_solution = True
        A_inv = np.linalg.inv(A_sys)
    else:
        x_sol = None
        has_solution = False
        A_inv = None

    # Visualization: Show as lines in 2D
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Geometric view (lines)
    _ax1 = axes3[0]
    _ax1.set_xlim(-5, 5)
    _ax1.set_ylim(-5, 5)
    _ax1.axhline(y=0, color='k', linewidth=0.5)
    _ax1.axvline(x=0, color='k', linewidth=0.5)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_aspect('equal')
    _ax1.set_title('Linear System as Intersecting Lines')
    _ax1.set_xlabel('x₁')
    _ax1.set_ylabel('x₂')

    _x_range = np.linspace(-5, 5, 100)

    # Line 1: a11*x1 + a12*x2 = b1  =>  x2 = (b1 - a11*x1) / a12
    if np.abs(A_sys[0, 1]) > 0.001:
        _y1 = (b_sys[0] - A_sys[0, 0] * _x_range) / A_sys[0, 1]
        _ax1.plot(_x_range, _y1, 'b-', linewidth=2, label=f'{A_sys[0,0]:.1f}x₁ + {A_sys[0,1]:.1f}x₂ = {b_sys[0]:.1f}')
    else:
        if np.abs(A_sys[0, 0]) > 0.001:
            _x_val = b_sys[0] / A_sys[0, 0]
            _ax1.axvline(x=_x_val, color='b', linewidth=2, label=f'x₁ = {_x_val:.1f}')

    # Line 2: a21*x1 + a22*x2 = b2  =>  x2 = (b2 - a21*x1) / a22
    if np.abs(A_sys[1, 1]) > 0.001:
        _y2 = (b_sys[1] - A_sys[1, 0] * _x_range) / A_sys[1, 1]
        _ax1.plot(_x_range, _y2, 'r-', linewidth=2, label=f'{A_sys[1,0]:.1f}x₁ + {A_sys[1,1]:.1f}x₂ = {b_sys[1]:.1f}')
    else:
        if np.abs(A_sys[1, 0]) > 0.001:
            _x_val = b_sys[1] / A_sys[1, 0]
            _ax1.axvline(x=_x_val, color='r', linewidth=2, label=f'x₁ = {_x_val:.1f}')

    # Plot solution point
    if has_solution:
        _ax1.plot(x_sol[0], x_sol[1], 'go', markersize=15, label=f'Solution: ({x_sol[0]:.2f}, {x_sol[1]:.2f})')

    _ax1.legend(loc='upper right', fontsize=9)
    _ax1.set_ylim(-5, 5)

    # Right: Summary
    _ax2 = axes3[1]
    _ax2.axis('off')

    if has_solution:
        summary = f"""
    Linear System Analysis
    ═══════════════════════════════════

    System: Ax = b

    A = [{A_sys[0,0]:6.2f}  {A_sys[0,1]:6.2f}]
        [{A_sys[1,0]:6.2f}  {A_sys[1,1]:6.2f}]

    b = [{b_sys[0]:6.2f}]
        [{b_sys[1]:6.2f}]

    ───────────────────────────────────

    Determinant: det(A) = {det_sys:.3f}

    Status: ✓ INVERTIBLE (unique solution exists)

    ───────────────────────────────────

    Solution: x = A⁻¹b

    A⁻¹ = [{A_inv[0,0]:6.3f}  {A_inv[0,1]:6.3f}]
          [{A_inv[1,0]:6.3f}  {A_inv[1,1]:6.3f}]

    x = [{x_sol[0]:6.3f}]
        [{x_sol[1]:6.3f}]

    ───────────────────────────────────

    Verification: Ax = [{A_sys@x_sol}]
                  b  = [{b_sys}]
        """
    else:
        summary = f"""
    Linear System Analysis
    ═══════════════════════════════════

    System: Ax = b

    A = [{A_sys[0,0]:6.2f}  {A_sys[0,1]:6.2f}]
        [{A_sys[1,0]:6.2f}  {A_sys[1,1]:6.2f}]

    b = [{b_sys[0]:6.2f}]
        [{b_sys[1]:6.2f}]

    ───────────────────────────────────

    Determinant: det(A) = {det_sys:.3f} ≈ 0

    Status: ✗ SINGULAR (not invertible)

    ───────────────────────────────────

    The system has either:
    • No solution (inconsistent), or
    • Infinitely many solutions

    The lines are parallel or coincident!

    ML Implication: Model is ill-conditioned.
    Use regularization or pseudo-inverse.
        """

    _ax2.text(0.05, 0.95, summary, transform=_ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue' if has_solution else 'lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. The Pseudo-Inverse: Handling Non-Invertible Cases

    When $\mathbf{A}$ is not square or not invertible, we use the **Moore-Penrose pseudo-inverse** $\mathbf{A}^+$:

    **Overdetermined system ($m > n$):** More equations than unknowns
    $$\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

    This gives the **least-squares solution**: minimizes $\|\mathbf{Ax} - \mathbf{b}\|_2^2$

    **Underdetermined system ($m < n$):** Fewer equations than unknowns
    $$\mathbf{A}^+ = \mathbf{A}^T(\mathbf{A}\mathbf{A}^T)^{-1}$$

    This gives the **minimum-norm solution**: smallest $\|\mathbf{x}\|_2$ among all solutions.
    """)
    return


@app.cell
def _(mo):
    # Overdetermined system demo
    n_points = mo.ui.slider(3, 10, value=5, step=1, label="Number of data points (m)")
    noise_level = mo.ui.slider(0, 2, value=0.5, step=0.1, label="Noise level")
    true_slope = mo.ui.slider(-2, 2, value=1.5, step=0.1, label="True slope")
    true_intercept = mo.ui.slider(-3, 3, value=0.5, step=0.1, label="True intercept")

    mo.md(f"""
    ### Least Squares: Fitting a Line to Noisy Data

    We'll fit $y = mx + c$ to noisy data using the pseudo-inverse.

    {n_points}
    {noise_level}
    {true_slope}
    {true_intercept}
    """)
    return n_points, noise_level, true_intercept, true_slope


@app.cell
def _(mo, n_points, noise_level, np, plt, true_intercept, true_slope):
    # Generate noisy data
    np.random.seed(42)
    m_pts = n_points.value
    x_data = np.linspace(0, 4, m_pts)
    y_true_line = true_slope.value * x_data + true_intercept.value
    y_data = y_true_line + noise_level.value * np.random.randn(m_pts)

    # Setup overdetermined system: [x, 1] @ [m, c]^T = y
    # A is m x 2, x is 2 x 1, b is m x 1
    A_ls = np.column_stack([x_data, np.ones(m_pts)])
    b_ls = y_data

    # Solve using pseudo-inverse (least squares)
    theta_ls = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]
    m_fit, c_fit = theta_ls

    # Predictions
    y_pred = A_ls @ theta_ls
    residuals = b_ls - y_pred
    mse = np.mean(residuals**2)

    # Visualization
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Data and fit
    _ax1 = axes4[0]
    _ax1.scatter(x_data, y_data, s=100, c='blue', alpha=0.7, label='Noisy data', zorder=3)
    _ax1.plot(x_data, y_true_line, 'g--', linewidth=2, label=f'True: y = {true_slope.value:.1f}x + {true_intercept.value:.1f}')
    _ax1.plot(x_data, y_pred, 'r-', linewidth=2, label=f'Fit: y = {m_fit:.2f}x + {c_fit:.2f}')

    # Draw residuals
    for _i in range(len(x_data)):
        _ax1.plot([x_data[_i], x_data[_i]], [y_data[_i], y_pred[_i]], 'gray', alpha=0.5)

    _ax1.set_xlabel('x')
    _ax1.set_ylabel('y')
    _ax1.set_title('Least Squares Fitting')
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    # Right: Residuals
    _ax2 = axes4[1]
    _ax2.bar(range(len(residuals)), residuals, color='orange', alpha=0.7)
    _ax2.axhline(y=0, color='k', linewidth=1)
    _ax2.set_xlabel('Data point index')
    _ax2.set_ylabel('Residual (y - ŷ)')
    _ax2.set_title(f'Residuals | MSE = {mse:.4f}')
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md(f"""
    ### Least Squares Solution via Pseudo-Inverse

    **Overdetermined system:** {m_pts} equations, 2 unknowns

    **True parameters:** slope = {true_slope.value:.2f}, intercept = {true_intercept.value:.2f}

    **Fitted parameters:** slope = {m_fit:.3f}, intercept = {c_fit:.3f}

    **Mean Squared Error:** {mse:.4f}

    **Interpretation:** The pseudo-inverse finds the line that minimizes the sum of squared residuals.
    This is the foundation of **linear regression** in machine learning!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary: Key Takeaways

    | Concept | Mathematical Form | ML Relevance |
    |---------|-------------------|--------------|
    | **Matrix as Transform** | $\mathbf{y} = \mathbf{Ax}$ | Every neural network layer |
    | **Determinant** | $\det(\mathbf{A})$ | Invertibility check, volume scaling |
    | **Matrix Inverse** | $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ | Solving exact systems |
    | **Pseudo-Inverse** | $\mathbf{A}^+$ | Least squares, underdetermined systems |
    | **Composition** | $\mathbf{AB} \neq \mathbf{BA}$ | Layer ordering in networks |

    ### Next Steps

    In the next notebook, we'll explore:
    - Eigenvalues and eigenvectors
    - Matrix diagonalization
    - Principal Component Analysis (PCA)

    ---

    *IME 775 - Mathematical Foundations of Deep Learning*
    """)
    return


if __name__ == "__main__":
    app.run()
