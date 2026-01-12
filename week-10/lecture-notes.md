# Week 10: Principles of Nonlinear Feature Engineering

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 10: Principles of Nonlinear Feature Engineering**.

---

## Overview

This week covers nonlinear feature transformations that extend linear models to capture complex patterns.

---

## Learning Objectives

- Understand limitations of linear models
- Apply polynomial and other nonlinear feature transformations
- Implement nonlinear regression and classification
- Understand the bias-variance trade-off

---

## 10.1 Introduction

### Linear Model Limitations

Linear models can only represent:
$$f(x) = w_0 + w_1 x_1 + \cdots + w_n x_n$$

This is a hyperplane—cannot capture curves or nonlinear patterns.

### The Key Insight

Transform features nonlinearly, then apply linear model:
$$f(x) = w^T \phi(x)$$

The model is **nonlinear in $x$** but **linear in $w$**!

---

## 10.2 Nonlinear Regression

### Polynomial Features

For single variable $x$:
$$\phi(x) = [1, x, x^2, \ldots, x^D]^T$$

Model:
$$f(x) = w_0 + w_1 x + w_2 x^2 + \cdots + w_D x^D$$

### Multivariate Polynomials

For $x = [x_1, x_2]^T$ with degree 2:
$$\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]^T$$

### Number of Features

For $n$ input dimensions and degree $D$:
$$\text{Number of features} = \binom{n + D}{D}$$

This grows quickly!

### Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit linear model on transformed features
model = LinearRegression()
model.fit(X_poly, y)
```

---

## 10.3 Nonlinear Multi-Output Regression

Same principle applies to multi-output:
$$f(x) = W^T \phi(x)$$

Where $W$ is a matrix of weights for each output.

---

## 10.4 Nonlinear Two-Class Classification

### Approach

1. Transform features: $\phi(x)$
2. Apply linear classifier: $\text{sign}(w^T \phi(x))$

### Example: XOR Problem

The XOR function is not linearly separable in $\mathbb{R}^2$.

But with feature $z = x_1 \cdot x_2$:
- Class 0: $x_1 x_2 > 0$ (both positive or both negative)
- Class 1: $x_1 x_2 < 0$ (different signs)

Now it's linearly separable!

### Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Polynomial features for classification
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Logistic regression
clf = LogisticRegression()
clf.fit(X_poly, y)
```

---

## 10.5 Nonlinear Multi-Class Classification

Same approach:
1. Transform features
2. Apply softmax or OvA classifier

---

## 10.6 Nonlinear Unsupervised Learning

### Kernel PCA

Apply PCA in a transformed feature space.

### Nonlinear Clustering

Use transformed features before clustering, or use kernel methods.

---

## Basis Functions

### General Form

$$f(x) = \sum_{j=1}^{M} w_j \phi_j(x)$$

### Common Basis Functions

| Type | Formula | Use Case |
|------|---------|----------|
| Polynomial | $x^j$ | Global patterns |
| Gaussian (RBF) | $\exp(-\frac{\|x - c_j\|^2}{2\sigma^2})$ | Local patterns |
| Sigmoid | $\sigma(a_j^T x + b_j)$ | Neural networks |
| Fourier | $\sin(j \omega x)$, $\cos(j \omega x)$ | Periodic patterns |

---

## Bias-Variance Trade-off

### Decomposition

$$\mathbb{E}[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

### Effect of Model Complexity

| Complexity | Bias | Variance | Total Error |
|------------|------|----------|-------------|
| Too low | High | Low | High (underfitting) |
| Just right | Medium | Medium | **Low** |
| Too high | Low | High | High (overfitting) |

### Controlling Complexity

1. **Feature degree**: Lower degree = simpler
2. **Number of features**: Fewer = simpler
3. **Regularization**: Higher λ = simpler

---

## Overfitting and Underfitting

### Underfitting (High Bias)

- Model too simple
- Cannot capture patterns
- High training error
- High test error

### Overfitting (High Variance)

- Model too complex
- Memorizes training data
- Low training error
- High test error

### Detecting Overfitting

Compare training and validation error:
- Large gap = overfitting
- Both high = underfitting

---

## Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Generate data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 30)).reshape(-1, 1)
y = np.sin(2 * np.pi * X.ravel()) + 0.3 * np.random.randn(30)

# Try different polynomial degrees
degrees = range(1, 15)
train_errors = []
cv_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    
    model = Ridge(alpha=0.001)
    model.fit(X_poly, y)
    
    train_error = np.mean((y - model.predict(X_poly))**2)
    cv_error = -cross_val_score(model, X_poly, y, 
                                 scoring='neg_mean_squared_error', cv=5).mean()
    
    train_errors.append(train_error)
    cv_errors.append(cv_error)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'b-o', label='Training Error')
plt.plot(degrees, cv_errors, 'r-o', label='CV Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Bias-Variance Trade-off')
plt.show()
```

---

## Exercises

### Exercise 10.1 (Section 10.2)
Fit polynomial regression models of degrees 1, 3, 5, 10 to noisy sine wave data. Plot and compare.

### Exercise 10.2 (Section 10.4)
Implement polynomial logistic regression for a circular classification boundary.

### Exercise 10.3
Demonstrate the bias-variance trade-off by computing training and test errors for different model complexities.

---

## Summary

- Nonlinear features enable linear models to capture complex patterns
- Polynomial features are common but grow exponentially
- Model complexity must be balanced (bias-variance trade-off)
- Cross-validation helps select appropriate complexity

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 10**.

### Supplementary Reading
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3.
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*, Chapter 7.

---

## Next Week Preview

**Week 11: Principles of Feature Learning & Cross-Validation** (Chapter 11)
- Universal approximators
- Cross-validation techniques
- Regularization for model selection
