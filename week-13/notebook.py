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
    # Week 13: Tree-Based Learners & Advanced Topics

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 14**

    ---

    ## Learning Objectives

    - Understand decision tree construction
    - Apply gradient boosting for improved performance
    - Implement random forests
    - Compare tree-based methods with other approaches
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    return (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        GradientBoostingClassifier,
        RandomForestClassifier,
        make_classification,
        np,
        plot_tree,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## From Stumps to Deep Trees (Section 14.2)

    ### Decision Stump

    A single split:
    $$f(x) = \begin{cases} c_1 & \text{if } x_j \leq t \\ c_2 & \text{if } x_j > t \end{cases}$$

    ### Deep Trees

    Recursively partition the feature space with more splits.

    ### Tree Building Algorithm

    ```
    1. If stopping criterion met, return leaf
    2. Find best split (feature j, threshold t)
    3. Split data into left/right
    4. Recursively build left and right subtrees
    ```
    """)
    return


@app.cell
def _(DecisionTreeClassifier, make_classification, np, plot_tree, plt):
    # Decision tree visualization
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                n_informative=2, n_clusters_per_class=1, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Decision boundary
    ax1 = axes[0]
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, edgecolors='black')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, edgecolors='black')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Decision Tree Boundaries (max_depth=3)')
    
    # Tree structure
    ax2 = axes[1]
    plot_tree(tree, ax=ax2, feature_names=['$x_1$', '$x_2$'], 
              class_names=['0', '1'], filled=True, rounded=True)
    ax2.set_title('Tree Structure')
    
    fig.suptitle('Decision Tree Classifier (ML Refined, Section 14.2)', fontsize=14)
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regression Trees (Section 14.3)

    ### Split Criterion

    Minimize squared error:
    $$\sum_{x_i \in R_L} (y_i - c_L)^2 + \sum_{x_i \in R_R} (y_i - c_R)^2$$

    Where $c_L = \text{mean}(y_i : x_i \in R_L)$ and $c_R = \text{mean}(y_i : x_i \in R_R)$.

    ### Prediction

    For a new point, traverse tree and return leaf value.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, np, plt):
    # Regression tree
    np.random.seed(42)
    X_reg = np.sort(np.random.uniform(0, 10, 100)).reshape(-1, 1)
    y_reg = np.sin(X_reg.ravel()) + 0.3 * np.random.randn(100)
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    depths = [1, 3, 10]
    
    for ax, depth in zip(axes2, depths):
        tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree_reg.fit(X_reg, y_reg)
        
        X_test = np.linspace(0, 10, 200).reshape(-1, 1)
        y_pred = tree_reg.predict(X_test)
        
        ax.scatter(X_reg, y_reg, alpha=0.7, s=20)
        ax.plot(X_test, y_pred, 'r-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'max_depth = {depth}')
        ax.grid(True, alpha=0.3)
    
    fig2.suptitle('Regression Trees with Different Depths', fontsize=14)
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Classification Trees (Section 14.4)

    ### Split Criteria

    | Criterion | Formula |
    |-----------|---------|
    | **Gini Impurity** | $\sum_c p_c(1 - p_c)$ |
    | **Entropy** | $-\sum_c p_c \log p_c$ |
    | **Misclassification** | $1 - \max_c p_c$ |

    Where $p_c$ is the proportion of class $c$ in the node.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient Boosting (Section 14.5)

    ### The Idea

    Build trees sequentially, each correcting errors of the ensemble:

    $$f_m(x) = f_{m-1}(x) + \gamma h_m(x)$$

    Where $h_m$ is fit to the **residuals** of $f_{m-1}$.

    ### Algorithm

    ```
    1. Initialize f₀(x) = mean(y)
    2. For m = 1 to M:
       a. Compute residuals: rᵢ = yᵢ - f_{m-1}(xᵢ)
       b. Fit tree hₘ to residuals
       c. Update: fₘ = f_{m-1} + γ·hₘ
    3. Return fₘ
    ```
    """)
    return


@app.cell
def _(GradientBoostingClassifier, X, np, plt, y):
    # Gradient Boosting learning curve
    from sklearn.model_selection import learning_curve
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        gb, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(train_sizes, train_scores.mean(axis=1), 'b-o', label='Training Score')
    ax3.plot(train_sizes, val_scores.mean(axis=1), 'r-o', label='Validation Score')
    ax3.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax3.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    ax3.set_xlabel('Training Set Size')
    ax3.set_ylabel('Score')
    ax3.set_title('Gradient Boosting Learning Curve (ML Refined, Section 14.5)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Random Forests (Section 14.6)

    ### The Idea

    Combine many trees trained on **random subsets** of data and features.

    ### Algorithm

    ```
    1. For b = 1 to B:
       a. Draw bootstrap sample
       b. Grow tree with random feature subset at each split
    2. Average predictions (regression) or vote (classification)
    ```

    ### Key Hyperparameters

    | Parameter | Effect |
    |-----------|--------|
    | `n_estimators` | More trees = less variance |
    | `max_features` | Controls correlation between trees |
    | `max_depth` | Controls individual tree complexity |
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    RandomForestClassifier,
    X,
    make_classification,
    np,
    plt,
    y,
):
    # Compare methods
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    X_compare, y_compare = make_classification(n_samples=500, n_features=20, 
                                                n_informative=10, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM (RBF)': SVC(kernel='rbf'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_compare, y_compare, cv=5)
        results[name] = (scores.mean(), scores.std())
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    means = [r[0] for r in results.values()]
    stds = [r[1] for r in results.values()]
    
    bars = ax4.bar(names, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['steelblue', 'coral', 'green', 'purple'])
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Model Comparison (5-Fold CV)')
    ax4.set_ylim(0.7, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{mean:.3f}', ha='center', fontsize=10)
    
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Pros | Cons |
    |--------|------|------|
    | **Decision Trees** | Interpretable, fast | Overfits easily |
    | **Random Forests** | Robust, parallel | Many trees needed |
    | **Gradient Boosting** | Often best accuracy | Sequential, slower |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 14.
    - **Supplementary**: Hastie, T. et al. (2009). *The Elements of Statistical Learning*, Chapters 9-10.

    ## Course Conclusion

    This completes the theoretical foundations of the course. Weeks 14-15 will focus on student presentations.
    """)
    return


if __name__ == "__main__":
    app.run()
