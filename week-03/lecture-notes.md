# Week 3: First-Order Optimization - Gradient Descent

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 3: First-Order Optimization Techniques**.

---

## Overview

This week introduces gradient descent, the foundational optimization algorithm for machine learning.

---

## Learning Objectives

- Understand the first-order optimality condition
- Derive and implement gradient descent
- Analyze learning rate selection
- Identify weaknesses of gradient descent

---

## 3.1 Introduction

First-order methods use **gradient information** to guide optimization:
- More efficient than zero-order methods
- Foundation of modern ML training
- Scalable to high dimensions

---

## 3.2 The First-Order Optimality Condition

### Necessary Condition for Optimality

For a differentiable function $g: \mathbb{R}^n \to \mathbb{R}$, if $w^*$ is a local minimum:

$$\nabla g(w^*) = 0$$

### The Gradient

$$\nabla g(w) = \begin{bmatrix} \frac{\partial g}{\partial w_1} \\ \vdots \\ \frac{\partial g}{\partial w_n} \end{bmatrix}$$

### Types of Stationary Points

| Type | Condition | Description |
|------|-----------|-------------|
| Local minimum | $\nabla g = 0$, $H \succ 0$ | Bowl-shaped |
| Local maximum | $\nabla g = 0$, $H \prec 0$ | Hilltop |
| Saddle point | $\nabla g = 0$, $H$ indefinite | Valley in one direction, peak in another |

Where $H$ is the Hessian matrix.

---

## 3.3 The Geometry of First-Order Taylor Series

### Taylor Expansion

Near a point $w$:
$$g(w + d) \approx g(w) + \nabla g(w)^T d$$

### Directional Derivative

The rate of change in direction $d$:
$$\frac{\partial g}{\partial d} = \nabla g(w)^T d = \|\nabla g(w)\| \|d\| \cos(\theta)$$

### Steepest Descent Direction

The direction that decreases $g$ most rapidly:
$$d^* = -\nabla g(w)$$

Because $\cos(\theta) = -1$ when $\theta = \pi$.

---

## 3.4 Computing Gradients Efficiently

### Analytical Gradients

For common functions:

| Function | Gradient |
|----------|----------|
| $g(w) = w^T w$ | $\nabla g = 2w$ |
| $g(w) = w^T A w$ | $\nabla g = (A + A^T)w$ |
| $g(w) = \|Xw - y\|^2$ | $\nabla g = 2X^T(Xw - y)$ |

### Automatic Differentiation

Libraries like autograd, JAX, PyTorch compute gradients automatically.

```python
import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
g = w[0]**2 + 3*w[1]**2
g.backward()
print(w.grad)  # tensor([2., 12.])
```

---

## 3.5 Gradient Descent

### The Update Rule

$$w^{(k+1)} = w^{(k)} - \alpha \nabla g(w^{(k)})$$

Where:
- $w^{(k)}$: Current iterate
- $\alpha > 0$: Learning rate (step size)
- $\nabla g(w^{(k)})$: Gradient at current point

### Algorithm

```python
def gradient_descent(g, grad_g, w0, alpha, max_iter, tol=1e-6):
    w = w0.copy()
    
    for k in range(max_iter):
        gradient = grad_g(w)
        
        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break
        
        # Update
        w = w - alpha * gradient
    
    return w
```

### Convergence Analysis

For **convex** functions with **L-Lipschitz** gradient:

$$g(w^{(k)}) - g(w^*) \leq \frac{\|w^{(0)} - w^*\|^2}{2\alpha k}$$

Convergence rate: $O(1/k)$ (sublinear)

For **strongly convex** functions:

$$g(w^{(k)}) - g(w^*) \leq \left(1 - \frac{\mu}{L}\right)^k (g(w^{(0)}) - g(w^*))$$

Convergence rate: Linear (exponential decay)

---

## Learning Rate Selection

### Impact of Learning Rate

| $\alpha$ | Behavior |
|----------|----------|
| $\alpha < 2/L$ | Convergence guaranteed |
| $\alpha = 1/L$ | Optimal for strongly convex |
| $\alpha > 2/L$ | May diverge |

### Practical Guidelines

1. **Start small**: Try $\alpha = 0.01$ or $0.001$
2. **Decay**: Reduce $\alpha$ over iterations
3. **Line search**: Adaptively choose $\alpha$ each step
4. **Learning rate schedules**: Step decay, cosine annealing

### Line Search Methods

**Backtracking (Armijo) line search:**
```python
def backtracking_line_search(g, grad_g, w, d, alpha=1.0, beta=0.5, c=1e-4):
    while g(w + alpha * d) > g(w) + c * alpha * grad_g(w).T @ d:
        alpha = beta * alpha
    return alpha
```

---

## 3.6 Two Natural Weaknesses of Gradient Descent

### Weakness 1: Zigzagging (Ill-Conditioning)

For functions with different curvature in different directions:
$$g(w) = w_1^2 + 100 w_2^2$$

The gradient points toward the minimum but the path zigzags.

**Condition number**: $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$

Large $\kappa$ → slow convergence.

### Weakness 2: Saddle Points

In high dimensions, saddle points are common:
- Gradient is zero but not a minimum
- GD can get stuck or slow down

### Solutions (Covered in Appendix A)

1. **Momentum**: Accumulate velocity to smooth path
2. **Adaptive learning rates**: RMSprop, Adam
3. **Second-order methods**: Use curvature information

---

## Implementation Examples

### Basic Gradient Descent

```python
import numpy as np

def gradient_descent(objective, gradient, x0, lr=0.01, n_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for _ in range(n_iter):
        grad = gradient(x)
        
        if np.linalg.norm(grad) < tol:
            break
            
        x = x - lr * grad
        history.append(x.copy())
    
    return x, np.array(history)

# Example: Minimize f(x) = x1^2 + 2*x2^2
def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

x_opt, history = gradient_descent(f, grad_f, [5.0, 3.0], lr=0.1)
print(f"Optimum: {x_opt}")
```

### Gradient Descent with Momentum

```python
def gradient_descent_momentum(gradient, x0, lr=0.01, momentum=0.9, n_iter=1000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    
    for _ in range(n_iter):
        grad = gradient(x)
        v = momentum * v + grad
        x = x - lr * v
    
    return x
```

### Gradient Descent for Linear Regression

```python
def linear_regression_gd(X, y, lr=0.01, n_iter=1000):
    n, p = X.shape
    w = np.zeros(p)
    
    for _ in range(n_iter):
        predictions = X @ w
        errors = predictions - y
        gradient = (2/n) * X.T @ errors
        w = w - lr * gradient
    
    return w

# Example
X = np.random.randn(100, 5)
true_w = np.array([1, 2, 3, 4, 5])
y = X @ true_w + 0.1 * np.random.randn(100)

w_learned = linear_regression_gd(X, y, lr=0.1, n_iter=1000)
print(f"True: {true_w}")
print(f"Learned: {w_learned}")
```

---

## Exercises

### Exercise 3.1 (Section 3.2)
Find all stationary points of $g(w_1, w_2) = w_1^2 + w_2^2 - 2w_1 - 4w_2 + 5$.

### Exercise 3.2 (Section 3.5)
Implement gradient descent to minimize $g(w) = (w - 3)^4$. Compare convergence for $\alpha = 0.01, 0.1, 0.5$.

### Exercise 3.3 (Section 3.6)
For $g(w_1, w_2) = w_1^2 + 100w_2^2$, run gradient descent from $(1, 1)$ with $\alpha = 0.01$. Plot the path and explain the zigzagging behavior.

---

## Summary

- First-order condition: $\nabla g(w^*) = 0$
- Gradient descent: $w \leftarrow w - \alpha \nabla g(w)$
- Learning rate is critical for convergence
- Weaknesses: zigzagging, saddle points

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 3**.

### Supplementary Reading
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Chapter 3.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv:1609.04747*.

---

## Next Week Preview

**Week 4: Second-Order Optimization: Newton's Method** (Chapter 4)
- Second-order optimality conditions
- Newton's method
- Hessian matrix
- Comparison with gradient descent
