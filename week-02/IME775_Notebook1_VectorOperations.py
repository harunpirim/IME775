"""
IME 775: Vector Operations - Interactive Exploration
=====================================================
A marimo notebook exploring fundamental vector operations in machine learning.

Course: IME 775 
Topics: Vectors, Dot Product, Cosine Similarity, L2 Norm
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
    # IME 775: Vector Operations in Machine Learning

    ## Learning Objectives

    1. Understand vectors as fundamental data structures in ML
    2. Explore dot product and its geometric meaning
    3. Visualize cosine similarity for measuring vector similarity
    4. Apply L2 norm to measure vector magnitude and error

    ---
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Vectors: The Building Blocks of ML

    A **vector** $\mathbf{x} \in \mathbb{R}^n$ is an ordered sequence of $n$ numbers:

    $$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

    In machine learning:
    - **Feature vectors** represent input data points
    - **Weight vectors** represent model parameters
    - **Output vectors** represent predictions (e.g., class probabilities)

    Let's create vectors interactively!
    """)
    return


@app.cell
def _(mo):
    # Interactive sliders for 2D vector components
    v1_x = mo.ui.slider(-5, 5, value=3, step=0.5, label="v₁ x-component")
    v1_y = mo.ui.slider(-5, 5, value=2, step=0.5, label="v₁ y-component")
    v2_x = mo.ui.slider(-5, 5, value=1, step=0.5, label="v₂ x-component")
    v2_y = mo.ui.slider(-5, 5, value=4, step=0.5, label="v₂ y-component")

    mo.md(f"""
    ### Create Your Vectors

    **Vector v₁:** {v1_x} {v1_y}

    **Vector v₂:** {v2_x} {v2_y}
    """)
    return v1_x, v1_y, v2_x, v2_y


@app.cell
def _(np, plt, v1_x, v1_y, v2_x, v2_y):
    # Create vectors
    v1 = np.array([v1_x.value, v1_y.value])
    v2 = np.array([v2_x.value, v2_y.value])

    # Calculate operations
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 > 0 and norm_v2 > 0:
        cos_sim = dot_product / (norm_v1 * norm_v2)
        _angle_rad = np.arccos(np.clip(cos_sim, -1, 1))
        angle_deg = np.degrees(_angle_rad)
    else:
        cos_sim = 0
        angle_deg = 0

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Vector visualization
    _ax1 = axes[0]
    _ax1.set_xlim(-6, 6)
    _ax1.set_ylim(-6, 6)
    _ax1.axhline(y=0, color='k', linewidth=0.5)
    _ax1.axvline(x=0, color='k', linewidth=0.5)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_aspect('equal')
    _ax1.set_xlabel('x')
    _ax1.set_ylabel('y')
    _ax1.set_title('Vector Visualization')

    # Draw vectors
    _ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.02, label=f'v₁ = [{v1[0]:.1f}, {v1[1]:.1f}]')
    _ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.02, label=f'v₂ = [{v2[0]:.1f}, {v2[1]:.1f}]')

    # Draw angle arc
    if norm_v1 > 0 and norm_v2 > 0:
        _theta1 = np.arctan2(v1[1], v1[0])
        _theta2 = np.arctan2(v2[1], v2[0])
        _arc_radius = min(norm_v1, norm_v2, 2) * 0.4
        _arc_angles = np.linspace(min(_theta1, _theta2), max(_theta1, _theta2), 50)
        _arc_x = _arc_radius * np.cos(_arc_angles)
        _arc_y = _arc_radius * np.sin(_arc_angles)
        _ax1.plot(_arc_x, _arc_y, 'g-', linewidth=2, label=f'θ = {angle_deg:.1f}°')

    _ax1.legend(loc='upper left')

    # Right plot: Results summary
    _ax2 = axes[1]
    _ax2.axis('off')

    results_text = f"""
    Vector Operations Results
    ═════════════════════════

    v₁ = [{v1[0]:.2f}, {v1[1]:.2f}]
    v₂ = [{v2[0]:.2f}, {v2[1]:.2f}]

    ─────────────────────────

    Dot Product: v₁ · v₂ = {dot_product:.3f}

    Formula: Σᵢ v₁ᵢ × v₂ᵢ
           = ({v1[0]:.2f} × {v2[0]:.2f}) + ({v1[1]:.2f} × {v2[1]:.2f})
           = {v1[0]*v2[0]:.2f} + {v1[1]*v2[1]:.2f}
           = {dot_product:.3f}

    ─────────────────────────

    L2 Norms:
    ‖v₁‖₂ = √({v1[0]:.2f}² + {v1[1]:.2f}²) = {norm_v1:.3f}
    ‖v₂‖₂ = √({v2[0]:.2f}² + {v2[1]:.2f}²) = {norm_v2:.3f}

    ─────────────────────────

    Cosine Similarity: cos(v₁, v₂) = {cos_sim:.3f}

    Angle between vectors: θ = {angle_deg:.1f}°

    ─────────────────────────

    Interpretation:
    {"• Vectors point in similar directions" if cos_sim > 0.5 else "• Vectors are roughly perpendicular" if -0.5 <= cos_sim <= 0.5 else "• Vectors point in opposite directions"}
    {"• Dot product > 0: acute angle" if dot_product > 0 else "• Dot product = 0: perpendicular" if dot_product == 0 else "• Dot product < 0: obtuse angle"}
    """

    _ax2.text(0.1, 0.95, results_text, transform=_ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. The Dot Product: Geometric Intuition

    The dot product has a beautiful geometric interpretation:

    $$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta$$

    This tells us:
    - **Magnitude matters**: Larger vectors → larger dot product
    - **Direction matters**: Aligned vectors → positive dot product
    - **Perpendicular = Zero**: Orthogonal vectors have zero dot product

    ### ML Application: Feature Similarity

    In document retrieval, each document is represented as a **TF-IDF vector**.
    The dot product measures how similar two documents are based on shared terms.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. L2 Norm: Measuring Vector Magnitude

    The L2 norm (Euclidean length) measures how "big" a vector is:

    $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

    ### ML Applications:

    1. **Error Measurement**: Mean Squared Error = $\frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$
    2. **Regularization**: L2 regularization penalizes large weights: $\|\mathbf{w}\|_2^2$
    3. **Normalization**: Unit vectors have $\|\mathbf{x}\|_2 = 1$

    Let's explore how vector magnitude affects learning!
    """)
    return


@app.cell
def _(mo):
    # Interactive exploration of L2 norm in error measurement
    y_true_1 = mo.ui.slider(0, 10, value=5, step=0.5, label="True value y₁")
    y_true_2 = mo.ui.slider(0, 10, value=8, step=0.5, label="True value y₂")
    y_pred_1 = mo.ui.slider(0, 10, value=4, step=0.5, label="Predicted ŷ₁")
    y_pred_2 = mo.ui.slider(0, 10, value=6, step=0.5, label="Predicted ŷ₂")

    mo.md(f"""
    ### Error Measurement with L2 Norm

    **True values:** {y_true_1} {y_true_2}

    **Predictions:** {y_pred_1} {y_pred_2}
    """)
    return y_pred_1, y_pred_2, y_true_1, y_true_2


@app.cell
def _(mo, np, plt, y_pred_1, y_pred_2, y_true_1, y_true_2):
    # Calculate error
    y_true = np.array([y_true_1.value, y_true_2.value])
    y_pred = np.array([y_pred_1.value, y_pred_2.value])
    error_vec = y_true - y_pred
    l2_error = np.linalg.norm(error_vec)
    mse = l2_error**2 / len(error_vec)

    # Visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Prediction vs True
    _ax1 = axes2[0]
    _indices = ['y₁', 'y₂']
    _x_pos = np.arange(len(_indices))
    _width = 0.35

    _bars1 = _ax1.bar(_x_pos - _width/2, y_true, _width, label='True values', color='green', alpha=0.7)
    _bars2 = _ax1.bar(_x_pos + _width/2, y_pred, _width, label='Predictions', color='orange', alpha=0.7)

    # Add error annotations
    for _i, (_yt, _yp) in enumerate(zip(y_true, y_pred)):
        _error = _yt - _yp
        _y_mid = (_yt + _yp) / 2
        _ax1.annotate(f'ε={_error:.1f}', xy=(_i, _y_mid), fontsize=10, ha='center',
                    color='red', fontweight='bold')

    _ax1.set_xticks(_x_pos)
    _ax1.set_xticklabels(_indices)
    _ax1.set_ylabel('Value')
    _ax1.set_title('True vs Predicted Values')
    _ax1.legend()
    _ax1.grid(axis='y', alpha=0.3)

    # Right plot: Error vector visualization
    _ax2 = axes2[1]
    _ax2.set_xlim(-5, 5)
    _ax2.set_ylim(-5, 5)
    _ax2.axhline(y=0, color='k', linewidth=0.5)
    _ax2.axvline(x=0, color='k', linewidth=0.5)
    _ax2.grid(True, alpha=0.3)
    _ax2.set_aspect('equal')

    # Draw error vector
    _ax2.quiver(0, 0, error_vec[0], error_vec[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.03, label=f'Error vector ε')

    # Draw circle showing L2 norm
    if l2_error > 0:
        _theta = np.linspace(0, 2*np.pi, 100)
        _ax2.plot(l2_error * np.cos(_theta), l2_error * np.sin(_theta),
                'b--', alpha=0.5, label=f'L2 norm = {l2_error:.2f}')

    _ax2.set_xlabel('ε₁ (error in y₁)')
    _ax2.set_ylabel('ε₂ (error in y₂)')
    _ax2.set_title('Error Vector in 2D Space')
    _ax2.legend()

    plt.tight_layout()

    mo.md(f"""
    ### Error Analysis Results

    | Metric | Formula | Value |
    |--------|---------|-------|
    | Error vector | ε = y - ŷ | [{error_vec[0]:.2f}, {error_vec[1]:.2f}] |
    | L2 Norm | ‖ε‖₂ = √(ε₁² + ε₂²) | **{l2_error:.3f}** |
    | MSE | ‖ε‖₂² / n | **{mse:.3f}** |

    **Interpretation:** The L2 norm represents the Euclidean distance between the true and predicted values in the output space.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Cosine Similarity: Direction Over Magnitude

    Cosine similarity measures how similar two vectors are, **regardless of their magnitudes**:

    $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

    **Range:** $[-1, 1]$
    - $+1$: Identical direction
    - $0$: Perpendicular (orthogonal)
    - $-1$: Opposite direction

    ### Why Cosine Similarity in ML?

    1. **Document Similarity**: Two documents about "machine learning" should be similar even if one is longer
    2. **Word Embeddings**: Word2Vec, GloVe use cosine similarity to find related words
    3. **Recommendation Systems**: Find users with similar preferences
    """)
    return


@app.cell
def _(mo):
    # Interactive demo: Document similarity
    doc1_ml = mo.ui.slider(0, 10, value=5, step=1, label="Doc1: 'ML' frequency")
    doc1_dl = mo.ui.slider(0, 10, value=4, step=1, label="Doc1: 'deep learning' frequency")
    doc2_ml = mo.ui.slider(0, 10, value=8, step=1, label="Doc2: 'ML' frequency")
    doc2_dl = mo.ui.slider(0, 10, value=6, step=1, label="Doc2: 'deep learning' frequency")

    mo.md(f"""
    ### Document Similarity Demo

    Imagine documents represented by term frequencies:

    **Document 1:** {doc1_ml} {doc1_dl}

    **Document 2:** {doc2_ml} {doc2_dl}
    """)
    return doc1_dl, doc1_ml, doc2_dl, doc2_ml


@app.cell
def _(doc1_dl, doc1_ml, doc2_dl, doc2_ml, mo, np, plt):
    # Calculate similarity
    doc1_vec = np.array([doc1_ml.value, doc1_dl.value])
    doc2_vec = np.array([doc2_ml.value, doc2_dl.value])

    norm1 = np.linalg.norm(doc1_vec)
    norm2 = np.linalg.norm(doc2_vec)

    if norm1 > 0 and norm2 > 0:
        cos_similarity = np.dot(doc1_vec, doc2_vec) / (norm1 * norm2)
        euclidean_dist = np.linalg.norm(doc1_vec - doc2_vec)
    else:
        cos_similarity = 0
        euclidean_dist = 0

    # Visualization
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Vector plot
    _ax1 = axes3[0]
    _max_val = max(doc1_ml.value, doc1_dl.value, doc2_ml.value, doc2_dl.value, 1) + 1
    _ax1.set_xlim(0, _max_val)
    _ax1.set_ylim(0, _max_val)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_aspect('equal')

    _ax1.quiver(0, 0, doc1_vec[0], doc1_vec[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.02, label=f'Doc1: [{doc1_vec[0]}, {doc1_vec[1]}]')
    _ax1.quiver(0, 0, doc2_vec[0], doc2_vec[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.02, label=f'Doc2: [{doc2_vec[0]}, {doc2_vec[1]}]')

    _ax1.set_xlabel("'Machine Learning' term frequency")
    _ax1.set_ylabel("'Deep Learning' term frequency")
    _ax1.set_title('Document Vectors in Term Space')
    _ax1.legend()

    # Right: Similarity metrics comparison
    _ax2 = axes3[1]
    _metrics = ['Cosine\nSimilarity', 'Euclidean\nDistance']
    _values = [cos_similarity, euclidean_dist]
    _colors = ['green' if cos_similarity > 0.8 else 'orange' if cos_similarity > 0.5 else 'red', 'blue']

    # Normalize for visualization
    _ax2_twin = _ax2.twinx()

    _ax2.bar([0], [cos_similarity], color=_colors[0], alpha=0.7, label='Cosine Similarity')
    _ax2.set_ylim(-1, 1)
    _ax2.set_ylabel('Cosine Similarity', color='green')
    _ax2.axhline(y=0, color='k', linewidth=0.5)

    _ax2_twin.bar([1], [euclidean_dist], color='blue', alpha=0.7, label='Euclidean Distance')
    _ax2_twin.set_ylabel('Euclidean Distance', color='blue')

    _ax2.set_xticks([0, 1])
    _ax2.set_xticklabels(_metrics)
    _ax2.set_title('Similarity Metrics Comparison')

    plt.tight_layout()

    similarity_level = "Very Similar" if cos_similarity > 0.9 else "Similar" if cos_similarity > 0.7 else "Somewhat Similar" if cos_similarity > 0.5 else "Different"

    mo.md(f"""
    ### Similarity Analysis

    | Document | Vector | Magnitude |
    |----------|--------|-----------|
    | Doc 1 | [{doc1_vec[0]}, {doc1_vec[1]}] | {norm1:.2f} |
    | Doc 2 | [{doc2_vec[0]}, {doc2_vec[1]}] | {norm2:.2f} |

    **Results:**
    - Cosine Similarity: **{cos_similarity:.3f}** → {similarity_level}
    - Euclidean Distance: **{euclidean_dist:.3f}**

    **Key Insight:** Cosine similarity ignores document length! Even if Doc2 is much longer (higher magnitude),
    if it discusses the same topics in similar proportions, the cosine similarity remains high.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Orthogonality: When Vectors are Independent

    Two vectors are **orthogonal** (perpendicular) when their dot product is zero:

    $$\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0$$

    ### Why Orthogonality Matters in ML:

    1. **Feature Independence**: Orthogonal features carry non-redundant information
    2. **PCA**: Principal components are orthogonal
    3. **Gram-Schmidt**: Creates orthonormal bases
    4. **Neural Networks**: Orthogonal weight initialization can improve training
    """)
    return


@app.cell
def _(mo):
    # Interactive orthogonality explorer
    orth_angle = mo.ui.slider(0, 180, value=90, step=5, label="Angle between vectors (degrees)")
    orth_mag = mo.ui.slider(0.5, 3, value=1.5, step=0.1, label="Vector magnitudes")

    mo.md(f"""
    ### Orthogonality Explorer

    Adjust the angle to see when vectors become orthogonal:

    {orth_angle}

    {orth_mag}
    """)
    return orth_angle, orth_mag


@app.cell
def _(mo, np, orth_angle, orth_mag, plt):
    # Create vectors at specified angle
    _angle_rad = np.radians(orth_angle.value)
    _mag = orth_mag.value

    _u = np.array([_mag, 0])
    _v = np.array([_mag * np.cos(_angle_rad), _mag * np.sin(_angle_rad)])

    _dot_prod = np.dot(_u, _v)
    _is_orthogonal = np.abs(_dot_prod) < 0.01

    _fig4, _ax = plt.subplots(figsize=(8, 8))
    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
    _ax.axhline(y=0, color='k', linewidth=0.5)
    _ax.axvline(x=0, color='k', linewidth=0.5)
    _ax.grid(True, alpha=0.3)
    _ax.set_aspect('equal')

    # Draw vectors
    _ax.quiver(0, 0, _u[0], _u[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.02, label=f'u = [{_u[0]:.2f}, {_u[1]:.2f}]')
    _ax.quiver(0, 0, _v[0], _v[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.02, label=f'v = [{_v[0]:.2f}, {_v[1]:.2f}]')

    # Highlight orthogonality
    if _is_orthogonal:
        # Draw right angle symbol
        _symbol_size = 0.3
        _ax.plot([_symbol_size, _symbol_size, 0], [0, _symbol_size, _symbol_size], 'g-', linewidth=2)
        _ax.set_title('✓ ORTHOGONAL: u · v = 0', fontsize=14, color='green', fontweight='bold')
    else:
        _ax.set_title(f'u · v = {_dot_prod:.3f}', fontsize=14)

    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.legend()

    plt.tight_layout()

    _status = "✓ **ORTHOGONAL**" if _is_orthogonal else "✗ Not orthogonal"

    mo.md(f"""
    ### Results

    - **Angle:** {orth_angle.value}°
    - **Dot Product:** u · v = {_dot_prod:.4f}
    - **Status:** {_status}

    **Observation:** At exactly 90°, the dot product becomes zero, indicating perfect orthogonality.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary: Key Takeaways

    | Operation | Formula | ML Application |
    |-----------|---------|----------------|
    | **Dot Product** | $\sum_i u_i v_i$ | Feature similarity, attention mechanisms |
    | **L2 Norm** | $\sqrt{\sum_i x_i^2}$ | Error measurement, regularization |
    | **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | Document retrieval, embeddings |
    | **Orthogonality** | $\mathbf{u} \cdot \mathbf{v} = 0$ | PCA, feature independence |

    ### Next Steps

    In the next notebook, we'll explore:
    - Matrix operations and transformations
    - Linear systems and their solutions
    - Eigendecomposition and its applications

    ---

    *IME 775 - Mathematical Foundations of Deep Learning*
    """)
    return


if __name__ == "__main__":
    app.run()
