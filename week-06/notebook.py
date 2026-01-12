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
    # Week 6: Linear Two-Class Classification

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 6**

    ---

    ## Learning Objectives

    - Understand logistic regression and cross-entropy loss
    - Implement the perceptron algorithm
    - Formulate and apply support vector machines
    - Evaluate classification quality metrics
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    return LogisticRegression, SVC, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction (Section 6.1)

    **Binary Classification**: Predict a discrete label $y \in \{-1, +1\}$ (or $\{0, 1\}$).

    ### The Linear Classifier

    $$\hat{y} = \text{sign}(w^T \tilde{x}) = \text{sign}(w_0 + w_1 x_1 + \cdots + w_n x_n)$$

    The decision boundary is a **hyperplane**: $w^T \tilde{x} = 0$
    """)
    return


@app.cell
def _(np, plt):
    # Visualize linear classification
    np.random.seed(42)
    n = 50
    
    # Generate two classes
    X_pos = np.random.randn(n, 2) + np.array([2, 2])
    X_neg = np.random.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*n + [-1]*n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=60, label='Class +1', edgecolors='black')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=60, label='Class -1', edgecolors='black')
    
    # Decision boundary
    x_line = np.linspace(-5, 5, 100)
    ax.plot(x_line, x_line, 'k-', linewidth=2, label='Decision boundary')
    ax.fill_between(x_line, x_line, 5, alpha=0.1, color='blue')
    ax.fill_between(x_line, x_line, -5, alpha=0.1, color='red')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Linear Binary Classification (ML Refined, Chapter 6)')
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Logistic Regression (Section 6.2)

    ### The Sigmoid Function

    $$\sigma(t) = \frac{1}{1 + e^{-t}}$$

    Maps any real number to $(0, 1)$ — interpretable as probability.

    ### The Model

    $$P(y = +1 | x) = \sigma(w^T \tilde{x}) = \frac{1}{1 + e^{-w^T \tilde{x}}}$$

    ### Cross-Entropy Cost

    $$g(w) = \frac{1}{P} \sum_{p=1}^{P} \log(1 + e^{-y_p w^T \tilde{x}_p})$$

    This is the **softmax cost** for binary classification.
    """)
    return


@app.cell
def _(np, plt):
    # Sigmoid function
    t = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-t))
    
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sigmoid
    ax1 = axes[0]
    ax1.plot(t, sigmoid, 'b-', linewidth=2)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('t')
    ax1.set_ylabel('σ(t)')
    ax1.set_title('Sigmoid Function σ(t) = 1/(1 + e⁻ᵗ)')
    ax1.grid(True, alpha=0.3)
    
    # Cross-entropy loss
    ax2 = axes[1]
    margin = np.linspace(-3, 3, 100)
    ce_loss = np.log(1 + np.exp(-margin))
    ax2.plot(margin, ce_loss, 'b-', linewidth=2, label='Cross-entropy')
    ax2.plot(margin, np.maximum(0, 1 - margin), 'r-', linewidth=2, label='Hinge (SVM)')
    ax2.set_xlabel('y · f(x) (margin)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Classification Loss Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Perceptron (Section 6.4)

    ### The Perceptron Cost

    $$g(w) = \frac{1}{P} \sum_{p=1}^{P} \max(0, -y_p w^T \tilde{x}_p)$$

    Only penalizes misclassified points.

    ### The Perceptron Algorithm

    For each misclassified point:
    $$w \leftarrow w + y_p \tilde{x}_p$$

    ### Properties

    - Converges for linearly separable data
    - Simple and fast
    - No probability output
    - Not unique solution
    """)
    return


@app.cell
def _(np, plt):
    # Perceptron algorithm
    np.random.seed(42)
    n_perc = 30
    
    X_pos_p = np.random.randn(n_perc, 2) + np.array([2, 2])
    X_neg_p = np.random.randn(n_perc, 2) + np.array([-2, -2])
    X_p = np.vstack([X_pos_p, X_neg_p])
    y_p = np.array([1]*n_perc + [-1]*n_perc)
    
    # Add bias
    X_aug = np.column_stack([np.ones(2*n_perc), X_p])
    
    # Perceptron
    w = np.zeros(3)
    history = [w.copy()]
    
    for epoch in range(10):
        for i in range(len(y_p)):
            if y_p[i] * (X_aug[i] @ w) <= 0:
                w = w + y_p[i] * X_aug[i]
                history.append(w.copy())
    
    # Visualization
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.scatter(X_pos_p[:, 0], X_pos_p[:, 1], c='blue', s=60, label='Class +1', edgecolors='black')
    ax3.scatter(X_neg_p[:, 0], X_neg_p[:, 1], c='red', s=60, label='Class -1', edgecolors='black')
    
    # Decision boundaries over time
    x_line = np.linspace(-5, 5, 100)
    for i, w_hist in enumerate(history[::max(1, len(history)//5)]):
        if abs(w_hist[2]) > 0.01:
            y_line = -(w_hist[0] + w_hist[1]*x_line) / w_hist[2]
            alpha = 0.3 + 0.7 * (i / len(history[::max(1, len(history)//5)]))
            ax3.plot(x_line, y_line, 'g-', alpha=alpha, linewidth=1)
    
    # Final boundary
    if abs(w[2]) > 0.01:
        y_line_final = -(w[0] + w[1]*x_line) / w[2]
        ax3.plot(x_line, y_line_final, 'k-', linewidth=2, label='Final boundary')
    
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title(f'Perceptron Learning ({len(history)} updates)')
    ax3.legend()
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)
    
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Support Vector Machines (Section 6.5)

    ### Maximum Margin Classifier

    Find the hyperplane that maximizes the **margin** — the distance to the nearest point.

    ### Optimization Problem

    $$\min_{w} \frac{1}{2}\|w\|^2$$

    Subject to:
    $$y_p(w^T \tilde{x}_p) \geq 1, \quad p = 1, \ldots, P$$

    ### Soft-Margin SVM

    For non-separable data:

    $$\min_{w,\xi} \frac{1}{2}\|w\|^2 + C \sum_{p=1}^{P} \xi_p$$

    Subject to:
    $$y_p(w^T \tilde{x}_p) \geq 1 - \xi_p, \quad \xi_p \geq 0$$
    """)
    return


@app.cell
def _(SVC, np, plt):
    # SVM visualization
    np.random.seed(42)
    n_svm = 50
    
    X_pos_s = np.random.randn(n_svm, 2) + np.array([1.5, 1.5])
    X_neg_s = np.random.randn(n_svm, 2) + np.array([-1.5, -1.5])
    X_s = np.vstack([X_pos_s, X_neg_s])
    y_s = np.array([1]*n_svm + [-1]*n_svm)
    
    # Fit SVM
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_s, y_s)
    
    # Create mesh
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Decision boundary and margins
    ax4.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                linestyles=['--', '-', '--'], linewidths=2)
    ax4.contourf(xx, yy, Z, levels=[-1, 1], alpha=0.2, colors=['gray'])
    
    ax4.scatter(X_pos_s[:, 0], X_pos_s[:, 1], c='blue', s=60, edgecolors='black')
    ax4.scatter(X_neg_s[:, 0], X_neg_s[:, 1], c='red', s=60, edgecolors='black')
    
    # Support vectors
    ax4.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidths=2,
                label=f'Support vectors ({len(svm.support_)})')
    
    ax4.set_xlabel('$x_1$')
    ax4.set_ylabel('$x_2$')
    ax4.set_title('Support Vector Machine (ML Refined, Section 6.5)')
    ax4.legend()
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Classification Quality Metrics (Section 6.8)

    ### Confusion Matrix

    |  | Predicted + | Predicted - |
    |--|------------|------------|
    | **Actual +** | True Positive (TP) | False Negative (FN) |
    | **Actual -** | False Positive (FP) | True Negative (TN) |

    ### Key Metrics

    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

    $$\text{Precision} = \frac{TP}{TP + FP}$$

    $$\text{Recall} = \frac{TP}{TP + FN}$$

    $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Cost Function | Properties |
    |--------|--------------|------------|
    | **Logistic Regression** | Cross-entropy | Probabilistic, smooth |
    | **Perceptron** | Hinge-like | Simple, fast |
    | **SVM** | Hinge + margin | Maximum margin, support vectors |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 6.
    - **Supplementary**: James, G. et al. (2023). *An Introduction to Statistical Learning*, Chapter 4.

    ## Next Week

    **Linear Multi-Class Classification** (Chapter 7): Extending to more than two classes.
    """)
    return


if __name__ == "__main__":
    app.run()
