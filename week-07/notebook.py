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
    # Week 7: Linear Multi-Class Classification

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 7**

    ---

    ## Learning Objectives

    - Extend binary classification to multiple classes
    - Implement one-versus-all (OvA) classification
    - Understand the softmax classifier
    - Apply multi-class quality metrics
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    return LogisticRegression, make_classification, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction (Section 7.1)

    **Multi-Class Classification**: Predict label $y \in \{1, 2, \ldots, C\}$ where $C > 2$.

    ### Examples

    - Digit recognition: $C = 10$ (0-9)
    - Object detection: Many categories
    - Disease diagnosis: Multiple conditions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## One-versus-All (OvA) Classification (Section 7.2)

    ### Strategy

    Train $C$ binary classifiers, one for each class:
    - Classifier $c$: Class $c$ vs all others

    ### Prediction

    $$\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} w_c^T \tilde{x}$$

    Choose the class with highest score.

    ### Training

    For each class $c$:
    - Positive examples: Class $c$
    - Negative examples: All other classes
    - Train binary classifier $w_c$
    """)
    return


@app.cell
def _(np, plt):
    # Generate 3-class data
    np.random.seed(42)
    n_per_class = 50
    
    # Three clusters
    X1 = np.random.randn(n_per_class, 2) + np.array([0, 3])
    X2 = np.random.randn(n_per_class, 2) + np.array([-3, -1])
    X3 = np.random.randn(n_per_class, 2) + np.array([3, -1])
    
    X = np.vstack([X1, X2, X3])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for c in range(3):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=60, 
                   label=f'Class {c}', edgecolors='black')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('3-Class Classification Problem (ML Refined, Chapter 7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig
    return


@app.cell
def _(LogisticRegression, X, np, plt, y):
    # One-vs-All classification
    from sklearn.preprocessing import label_binarize
    
    # Train OvA classifier
    clf = LogisticRegression(multi_class='ovr', max_iter=1000)
    clf.fit(X, y)
    
    # Create decision boundary mesh
    xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-4, 6, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax2.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    colors = ['blue', 'red', 'green']
    for c in range(3):
        mask = y == c
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=60, 
                   label=f'Class {c}', edgecolors='black')
    
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('One-vs-All Decision Boundaries')
    ax2.legend()
    
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Softmax Classifier (Section 7.5)

    ### The Softmax Function

    $$P(y = c | x) = \frac{e^{w_c^T \tilde{x}}}{\sum_{j=1}^{C} e^{w_j^T \tilde{x}}}$$

    ### Properties

    - Outputs valid probabilities (sum to 1)
    - Generalizes logistic regression to $C$ classes
    - All classifiers trained jointly

    ### Cross-Entropy Cost

    $$g(W) = -\frac{1}{P} \sum_{p=1}^{P} \log\left(\frac{e^{w_{y_p}^T \tilde{x}_p}}{\sum_{c=1}^{C} e^{w_c^T \tilde{x}_p}}\right)$$
    """)
    return


@app.cell
def _(np, plt):
    # Softmax visualization
    def softmax(z):
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / exp_z.sum()
    
    # Example
    z = np.array([2.0, 1.0, 0.1])
    probs = softmax(z)
    
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax1.bar(['z₁=2.0', 'z₂=1.0', 'z₃=0.1'], z, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Raw Score')
    ax1.set_title('Input Scores')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar(['P(c=1)', 'P(c=2)', 'P(c=3)'], probs, color='coral', alpha=0.7)
    ax2.set_ylabel('Probability')
    ax2.set_title('Softmax Output')
    for i, p in enumerate(probs):
        ax2.text(i, p + 0.02, f'{p:.2f}', ha='center')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Multi-Class Quality Metrics (Section 7.6)

    ### Confusion Matrix

    For $C$ classes, a $C \times C$ matrix where entry $(i, j)$ counts predictions of class $j$ when true class is $i$.

    ### Per-Class Metrics

    For each class $c$:
    - **Precision**: $\frac{TP_c}{TP_c + FP_c}$
    - **Recall**: $\frac{TP_c}{TP_c + FN_c}$

    ### Averaging Strategies

    | Method | Description |
    |--------|-------------|
    | **Macro** | Average per-class metrics |
    | **Micro** | Compute globally across all classes |
    | **Weighted** | Weight by class frequency |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Description | Use Case |
    |--------|-------------|----------|
    | **One-vs-All** | $C$ binary classifiers | Simple, independent training |
    | **Softmax** | Joint probability model | Calibrated probabilities |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 7.
    - **Supplementary**: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.

    ## Next Week

    **Linear Unsupervised Learning & PCA** (Chapter 8): Dimensionality reduction and clustering.
    """)
    return


if __name__ == "__main__":
    app.run()
