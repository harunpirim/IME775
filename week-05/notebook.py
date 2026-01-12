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
    # Week 5: Linear Regression

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 5**

    ---

    ## Learning Objectives

    - Formulate and solve least squares linear regression
    - Understand least absolute deviations regression
    - Apply regression quality metrics
    - Implement weighted regression
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    return LinearRegression, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction (Section 5.1)

    **Linear Regression**: Learn a linear relationship between input features and a continuous output.

    ### The Model

    $$f(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = w^T \tilde{x}$$

    Where $\tilde{x} = [1, x_1, \ldots, x_n]^T$ includes the bias term.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Least Squares Linear Regression (Section 5.2)

    ### The Cost Function

    $$g(w) = \frac{1}{P} \sum_{p=1}^{P} (y_p - f(x_p))^2 = \frac{1}{P} \|y - Xw\|^2$$

    Where:
    - $P$: Number of training points
    - $y$: Vector of outputs
    - $X$: Design matrix (rows are $\tilde{x}_p^T$)
    - $w$: Weight vector

    ### The Normal Equations

    Setting $\nabla g = 0$:

    $$X^T X w = X^T y$$

    Solution:

    $$w^* = (X^T X)^{-1} X^T y$$
    """)
    return


@app.cell
def _(np, plt):
    # Generate data
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 10, n)
    y_true = 2.5 * x + 3
    y = y_true + np.random.randn(n) * 2
    
    # Build design matrix
    X = np.column_stack([np.ones(n), x])
    
    # Solve normal equations
    w = np.linalg.solve(X.T @ X, X.T @ y)
    y_pred = X @ w
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.7, s=50, label='Data')
    ax.plot(x, y_pred, 'r-', linewidth=2, 
            label=f'Fit: y = {w[1]:.2f}x + {w[0]:.2f}')
    ax.plot(x, y_true, 'g--', linewidth=1, alpha=0.7, label='True: y = 2.5x + 3')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Least Squares Linear Regression (ML Refined, Section 5.2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient Descent for Linear Regression

    The gradient of the least squares cost:

    $$\nabla g(w) = \frac{2}{P} X^T(Xw - y)$$

    ### Gradient Descent Update

    $$w^{(k+1)} = w^{(k)} - \alpha \cdot \frac{2}{P} X^T(Xw^{(k)} - y)$$
    """)
    return


@app.cell
def _(np, plt):
    # Gradient descent for linear regression
    np.random.seed(42)
    n_gd = 100
    x_gd = np.random.randn(n_gd, 2)
    true_w = np.array([3, -2, 1])  # bias, w1, w2
    X_gd = np.column_stack([np.ones(n_gd), x_gd])
    y_gd = X_gd @ true_w + 0.5 * np.random.randn(n_gd)
    
    # Gradient descent
    def gradient(w, X, y):
        return (2/len(y)) * X.T @ (X @ w - y)
    
    w = np.zeros(3)
    history = [w.copy()]
    lr = 0.1
    
    for _ in range(50):
        w = w - lr * gradient(w, X_gd, y_gd)
        history.append(w.copy())
    
    history = np.array(history)
    
    # Closed-form solution
    w_closed = np.linalg.solve(X_gd.T @ X_gd, X_gd.T @ y_gd)
    
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parameter convergence
    ax1 = axes[0]
    for i, label in enumerate(['$w_0$', '$w_1$', '$w_2$']):
        ax1.plot(history[:, i], label=f'{label} (true={true_w[i]})')
        ax1.axhline(true_w[i], linestyle='--', alpha=0.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Parameter value')
    ax1.set_title('Parameter Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost convergence
    ax2 = axes[1]
    costs = [np.mean((X_gd @ h - y_gd)**2) for h in history]
    ax2.plot(costs, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE Cost')
    ax2.set_title('Cost Function Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Least Absolute Deviations (Section 5.3)

    An alternative to least squares:

    $$g(w) = \frac{1}{P} \sum_{p=1}^{P} |y_p - f(x_p)|$$

    ### Comparison

    | Aspect | Least Squares | Least Absolute Deviations |
    |--------|--------------|--------------------------|
    | Cost | $(y - \hat{y})^2$ | $|y - \hat{y}|$ |
    | Outlier sensitivity | High | Low |
    | Closed-form | Yes | No |
    | Optimization | Easy | Requires iterative methods |

    ### When to Use LAD

    - Data contains outliers
    - Heavy-tailed error distribution
    - Robust estimation required
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regression Quality Metrics (Section 5.4)

    ### Mean Squared Error (MSE)

    $$\text{MSE} = \frac{1}{P} \sum_{p=1}^{P} (y_p - \hat{y}_p)^2$$

    ### Root Mean Squared Error (RMSE)

    $$\text{RMSE} = \sqrt{\text{MSE}}$$

    ### Mean Absolute Error (MAE)

    $$\text{MAE} = \frac{1}{P} \sum_{p=1}^{P} |y_p - \hat{y}_p|$$

    ### Coefficient of Determination ($R^2$)

    $$R^2 = 1 - \frac{\sum_p (y_p - \hat{y}_p)^2}{\sum_p (y_p - \bar{y})^2}$$

    - $R^2 = 1$: Perfect fit
    - $R^2 = 0$: Model is no better than predicting the mean
    - $R^2 < 0$: Model is worse than predicting the mean
    """)
    return


@app.cell
def _(X, np, w, y):
    # Calculate metrics
    y_pred_metrics = X @ w
    residuals = y - y_pred_metrics
    
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    
    print("Regression Quality Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weighted Regression (Section 5.5)

    When data points have different importance or reliability:

    $$g(w) = \frac{1}{P} \sum_{p=1}^{P} \beta_p (y_p - f(x_p))^2$$

    Where $\beta_p > 0$ is the weight for point $p$.

    ### Normal Equations for Weighted Regression

    $$X^T B X w = X^T B y$$

    Where $B = \text{diag}(\beta_1, \ldots, \beta_P)$.

    ### Applications

    - Heteroscedastic data (varying error variance)
    - Importance weighting
    - Sample weighting for imbalanced data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Key Points |
    |---------|------------|
    | **Least Squares** | Minimize squared errors, closed-form solution |
    | **LAD** | Minimize absolute errors, robust to outliers |
    | **Metrics** | MSE, RMSE, MAE, $R^2$ |
    | **Weighted** | Different importance for different points |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 5.
    - **Supplementary**: James, G. et al. (2023). *An Introduction to Statistical Learning*, Chapter 3.

    ## Next Week

    **Linear Two-Class Classification** (Chapter 6): Logistic regression, perceptron, and SVM.
    """)
    return


if __name__ == "__main__":
    app.run()
