# Week 5: Linear Regression

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 5: Linear Regression**.

---

## Overview

This week covers linear regression from the optimization perspective, including least squares, robust regression, and quality metrics.

---

## Learning Objectives

- Formulate and solve least squares linear regression
- Understand least absolute deviations for robust regression
- Apply regression quality metrics
- Implement weighted regression

---

## 5.1 Introduction

### The Regression Problem

Given training data $\{(x_p, y_p)\}_{p=1}^P$:
- $x_p \in \mathbb{R}^n$: Input features
- $y_p \in \mathbb{R}$: Continuous output

**Goal**: Learn a function $f(x)$ that predicts $y$ from $x$.

### Linear Model

$$f(x) = w_0 + w_1 x_1 + \cdots + w_n x_n = w^T \tilde{x}$$

Where:
- $\tilde{x} = [1, x_1, \ldots, x_n]^T$: Augmented input (includes bias)
- $w = [w_0, w_1, \ldots, w_n]^T$: Weight vector

---

## 5.2 Least Squares Linear Regression

### The Cost Function

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} (y_p - w^T \tilde{x}_p)^2$$

In matrix form:
$$g(w) = \frac{1}{P} \|y - Xw\|_2^2$$

Where:
- $X \in \mathbb{R}^{P \times (n+1)}$: Design matrix (rows are $\tilde{x}_p^T$)
- $y \in \mathbb{R}^P$: Output vector

### The Normal Equations

Taking the gradient and setting to zero:

$$\nabla_w g = \frac{2}{P} X^T(Xw - y) = 0$$

Solving:
$$X^T X w = X^T y$$

**Closed-form solution:**
$$w^* = (X^T X)^{-1} X^T y$$

### Gradient Descent Alternative

When $X^T X$ is ill-conditioned or problem is large:

$$w^{(k+1)} = w^{(k)} - \alpha \cdot \frac{2}{P} X^T(Xw^{(k)} - y)$$

### Implementation

```python
import numpy as np

# Method 1: Normal equations
def least_squares_closed(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# Method 2: Gradient descent
def least_squares_gd(X, y, lr=0.01, n_iter=1000):
    w = np.zeros(X.shape[1])
    P = len(y)
    
    for _ in range(n_iter):
        gradient = (2/P) * X.T @ (X @ w - y)
        w = w - lr * gradient
    
    return w

# Example
X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
true_w = np.array([1, 2, 3])
y = X @ true_w + 0.1 * np.random.randn(100)

w_closed = least_squares_closed(X, y)
w_gd = least_squares_gd(X, y, lr=0.1)
print(f"Closed form: {w_closed}")
print(f"GD: {w_gd}")
```

---

## 5.3 Least Absolute Deviations

### The Cost Function

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} |y_p - w^T \tilde{x}_p|$$

### Why LAD?

- **Robust to outliers**: Large errors contribute linearly, not quadratically
- **Median-like behavior**: Less sensitive to extreme values

### Comparison with Least Squares

| Aspect | Least Squares | LAD |
|--------|--------------|-----|
| Error term | $(y - \hat{y})^2$ | $|y - \hat{y}|$ |
| Outlier sensitivity | High | Low |
| Closed-form solution | Yes | No |
| Derivative | Smooth | Not smooth at 0 |
| Estimation | Mean | Median |

### Optimization

LAD requires iterative methods:
- Subgradient descent
- Iteratively reweighted least squares (IRLS)
- Linear programming

```python
from scipy.optimize import minimize

def lad_cost(w, X, y):
    return np.mean(np.abs(y - X @ w))

result = minimize(lad_cost, x0=np.zeros(X.shape[1]), 
                  args=(X, y), method='BFGS')
w_lad = result.x
```

---

## 5.4 Regression Quality Metrics

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{P} \sum_{p=1}^{P} (y_p - \hat{y}_p)^2$$

- Same units as $y^2$
- Penalizes large errors heavily

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Same units as $y$
- More interpretable than MSE

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{P} \sum_{p=1}^{P} |y_p - \hat{y}_p|$$

- Same units as $y$
- Robust to outliers

### R-squared (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_p (y_p - \hat{y}_p)^2}{\sum_p (y_p - \bar{y})^2}$$

**Interpretation:**
- $R^2 = 1$: Perfect prediction
- $R^2 = 0$: Model predicts the mean
- $R^2 < 0$: Model is worse than predicting the mean

### Implementation

```python
def regression_metrics(y_true, y_pred):
    n = len(y_true)
    residuals = y_true - y_pred
    
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
```

---

## 5.5 Weighted Regression

### Motivation

Not all data points are equally reliable or important:
- Different measurement noise
- Different sample sizes
- Importance weighting

### Weighted Least Squares Cost

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} \beta_p (y_p - w^T \tilde{x}_p)^2$$

Where $\beta_p > 0$ is the weight for point $p$.

### Matrix Form

$$g(w) = \frac{1}{P} (y - Xw)^T B (y - Xw)$$

Where $B = \text{diag}(\beta_1, \ldots, \beta_P)$.

### Normal Equations

$$X^T B X w = X^T B y$$

**Solution:**
$$w^* = (X^T B X)^{-1} X^T B y$$

### Applications

1. **Heteroscedastic data**: $\beta_p = 1/\sigma_p^2$
2. **Sample weighting**: Give more weight to important samples
3. **Iteratively reweighted least squares**: Use for robust regression

---

## 5.6 Multi-Output Regression

### Problem Setup

Multiple outputs: $y_p \in \mathbb{R}^C$

$$Y = XW$$

Where $W \in \mathbb{R}^{(n+1) \times C}$.

### Solution

Solve independently for each output, or jointly:

$$W^* = (X^T X)^{-1} X^T Y$$

---

## Regularization (Preview)

To prevent overfitting:

### Ridge Regression (L2)

$$g(w) = \frac{1}{P} \|y - Xw\|^2 + \lambda \|w\|_2^2$$

### Lasso (L1)

$$g(w) = \frac{1}{P} \|y - Xw\|^2 + \lambda \|w\|_1$$

*Covered in more detail in Chapter 9.*

---

## Exercises

### Exercise 5.1 (Section 5.2)
Derive the normal equations from the gradient of the least squares cost function.

### Exercise 5.2 (Section 5.3)
Generate data with outliers. Compare least squares vs LAD regression.

### Exercise 5.3 (Section 5.4)
For a given model, compute MSE, RMSE, MAE, and $R^2$. How do they relate?

### Exercise 5.4 (Section 5.5)
Implement weighted least squares and apply it to data with heteroscedastic noise.

---

## Summary

- Least squares minimizes squared errors, has closed-form solution
- LAD is more robust but requires iterative optimization
- Quality metrics: MSE, RMSE, MAE, R²
- Weighted regression handles varying reliability

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 5**.

### Supplementary Reading
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An Introduction to Statistical Learning* (2nd ed.), Chapter 3.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 3.

---

## Next Week Preview

**Week 6: Linear Two-Class Classification** (Chapter 6)
- Logistic regression
- The perceptron
- Support vector machines
- Classification metrics
