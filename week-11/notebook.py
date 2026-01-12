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
    # Week 11: Principles of Feature Learning & Cross-Validation

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 11**

    ---

    ## Learning Objectives

    - Understand universal approximation
    - Apply cross-validation for model selection
    - Implement regularization strategies
    - Distinguish training, validation, and test data
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    return KFold, PolynomialFeatures, Ridge, cross_val_score, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Universal Approximators (Section 11.2)

    ### The Goal

    Find functions that can approximate **any** continuous function arbitrarily well.

    ### Examples of Universal Approximators

    | Type | Form |
    |------|------|
    | Polynomials | $\sum_j w_j x^j$ |
    | Neural Networks | Multi-layer perceptrons |
    | Kernel Methods | $\sum_j w_j K(x, x_j)$ |
    | Trees | Ensemble of decision trees |

    ### The Catch

    Universal approximation ≠ learning from finite data!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Naive Cross-Validation (Section 11.4)

    ### The Problem

    How do we choose model complexity (e.g., polynomial degree)?

    ### Train-Validation Split

    1. Split data: Training set + Validation set
    2. Train model on training set
    3. Evaluate on validation set
    4. Choose complexity that minimizes validation error

    ### The Danger

    Using test data for model selection leads to **overfitting** to the test set!
    """)
    return


@app.cell
def _(PolynomialFeatures, Ridge, np, plt):
    # Cross-validation example
    np.random.seed(42)
    n = 50
    X = np.sort(np.random.uniform(0, 1, n)).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel()) + 0.3 * np.random.randn(n)
    
    # Split into train/validation
    train_idx = np.random.choice(n, int(0.7*n), replace=False)
    val_idx = np.setdiff1d(np.arange(n), train_idx)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    degrees = range(1, 15)
    train_errors = []
    val_errors = []
    
    for d in degrees:
        poly = PolynomialFeatures(degree=d)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        model = Ridge(alpha=0.0001)
        model.fit(X_train_poly, y_train)
        
        train_errors.append(np.mean((y_train - model.predict(X_train_poly))**2))
        val_errors.append(np.mean((y_val - model.predict(X_val_poly))**2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(degrees, train_errors, 'b-o', label='Training Error')
    ax.plot(degrees, val_errors, 'r-o', label='Validation Error')
    ax.axvline(degrees[np.argmin(val_errors)], color='g', linestyle='--', 
               label=f'Best degree: {degrees[np.argmin(val_errors)]}')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Train-Validation Split (ML Refined, Section 11.4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## K-Fold Cross-Validation (Section 11.10)

    ### Why K-Fold?

    - More robust than single split
    - Uses all data for both training and validation
    - Less variance in estimate

    ### Algorithm

    ```
    1. Split data into K folds
    2. For k = 1 to K:
       a. Train on all folds except k
       b. Validate on fold k
    3. Average the K validation errors
    ```

    ### Common Choices

    - K = 5 or K = 10 (standard)
    - K = n (Leave-One-Out, high variance)
    """)
    return


@app.cell
def _(KFold, PolynomialFeatures, Ridge, X, np, plt, y):
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    degrees_kfold = range(1, 12)
    cv_mean_errors = []
    cv_std_errors = []
    
    for d in degrees_kfold:
        poly = PolynomialFeatures(degree=d)
        fold_errors = []
        
        for train_idx_kf, val_idx_kf in kfold.split(X):
            X_train_kf = poly.fit_transform(X[train_idx_kf])
            X_val_kf = poly.transform(X[val_idx_kf])
            
            model = Ridge(alpha=0.0001)
            model.fit(X_train_kf, y[train_idx_kf])
            
            error = np.mean((y[val_idx_kf] - model.predict(X_val_kf))**2)
            fold_errors.append(error)
        
        cv_mean_errors.append(np.mean(fold_errors))
        cv_std_errors.append(np.std(fold_errors))
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(degrees_kfold, cv_mean_errors, yerr=cv_std_errors, 
                 fmt='b-o', capsize=3, label='5-Fold CV Error')
    ax2.axvline(list(degrees_kfold)[np.argmin(cv_mean_errors)], color='g', linestyle='--',
                label=f'Best degree: {list(degrees_kfold)[np.argmin(cv_mean_errors)]}')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('K-Fold Cross-Validation (ML Refined, Section 11.10)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Efficient Cross-Validation via Regularization (Section 11.6)

    ### The Idea

    Instead of varying model complexity, fix complexity and vary regularization.

    $$\min_w \|y - X_\phi w\|^2 + \lambda \|w\|^2$$

    - Small $\lambda$: Complex model (low bias, high variance)
    - Large $\lambda$: Simple model (high bias, low variance)

    ### Advantages

    - Continuous hyperparameter
    - Often faster to tune
    - Built-in with many libraries
    """)
    return


@app.cell
def _(PolynomialFeatures, Ridge, X, np, plt, y):
    # Regularization path
    poly_reg = PolynomialFeatures(degree=10)
    X_poly = poly_reg.fit_transform(X)
    
    alphas = np.logspace(-6, 2, 50)
    
    # Store coefficients
    coefs = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_poly, y)
        coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i in range(1, min(10, coefs.shape[1])):
        ax3.semilogx(alphas, coefs[:, i], linewidth=2, label=f'$w_{i}$')
    
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_xlabel('Regularization parameter λ')
    ax3.set_ylabel('Coefficient value')
    ax3.set_title('Regularization Path (ML Refined, Section 11.6)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Testing Data (Section 11.7)

    ### The Three-Way Split

    | Set | Purpose | Used For |
    |-----|---------|----------|
    | **Training** | Fit model | Learning parameters |
    | **Validation** | Model selection | Hyperparameter tuning |
    | **Test** | Final evaluation | Report performance |

    ### Critical Rule

    > **Never use test data for any decision-making!**

    Test data should only be touched once, at the very end.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Key Point |
    |---------|-----------|
    | **Universal approximators** | Can fit any function (with enough capacity) |
    | **Cross-validation** | Estimate generalization error |
    | **K-Fold CV** | More robust than single split |
    | **Regularization** | Control complexity continuously |
    | **Test set** | Only for final evaluation |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 11.
    - **Supplementary**: Hastie, T. et al. (2009). *The Elements of Statistical Learning*, Chapter 7.

    ## Next Week

    **Kernel Methods & Neural Networks** (Chapters 12-13): Advanced nonlinear models.
    """)
    return


if __name__ == "__main__":
    app.run()
