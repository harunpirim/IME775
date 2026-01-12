# Week 2: Calculus Foundations for Deep Learning

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 3: Calculus for Deep Learning**.

---

## Overview

This module covers the calculus fundamentals required for understanding how neural networks learn through gradient-based optimization.

---

## Learning Objectives

- Master derivatives and their role in optimization
- Understand multivariable calculus for neural networks
- Learn the chain rule as the foundation of backpropagation
- Connect gradients to learning algorithms

---

## 2.1 Derivatives: Measuring Change

### Single Variable Derivative

The derivative measures instantaneous rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

### Geometric Interpretation

- Slope of tangent line
- Direction of steepest ascent (1D)

### Common Derivatives

| Function | Derivative | Deep Learning Context |
|----------|------------|----------------------|
| $x^n$ | $nx^{n-1}$ | Polynomial features |
| $e^x$ | $e^x$ | Exponential activations |
| $\ln(x)$ | $1/x$ | Log-likelihood |
| $\sin(x)$ | $\cos(x)$ | Positional encodings |
| $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid derivative |

---

## 2.2 Partial Derivatives

### Definition

For function $f(x_1, x_2, \ldots, x_n)$:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### Example

For $f(x, y) = x^2 + xy + y^2$:
- $\frac{\partial f}{\partial x} = 2x + y$
- $\frac{\partial f}{\partial y} = x + 2y$

### Interpretation

Partial derivative measures how function changes when varying one variable while holding others fixed.

---

## 2.3 The Gradient

### Definition

For scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Properties

1. **Direction**: Points toward steepest ascent
2. **Magnitude**: Rate of maximum increase
3. **Perpendicular**: To level sets of $f$

### Gradient in Neural Networks

If $L(\theta)$ is the loss function:
- $\nabla_\theta L$: Direction to move parameters to increase loss
- $-\nabla_\theta L$: Direction to move parameters to **decrease** loss

---

## 2.4 The Chain Rule

### Single Variable

If $y = f(g(x))$, then:
$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### Multivariable Chain Rule

If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:
$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

### General Form

$$\frac{\partial L}{\partial \theta} = \sum_{\text{paths}} \prod_{\text{edges}} \frac{\partial}{\partial}$$

**This is the foundation of backpropagation!**

---

## 2.5 The Jacobian Matrix

### Definition

For vector-valued function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$:

$$\mathbf{J} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

### Layer Jacobian

For layer $\mathbf{y} = \mathbf{f}(\mathbf{x})$:
- $\mathbf{J} \in \mathbb{R}^{m \times n}$
- Entry $(i, j)$ tells how output $i$ changes with input $j$

### Chain Rule with Jacobians

$$\mathbf{J}_{composite} = \mathbf{J}_f \cdot \mathbf{J}_g$$

---

## 2.6 The Hessian Matrix

### Definition

Second-order partial derivatives of scalar $f$:

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

### Properties

- Symmetric (for smooth functions)
- Eigenvalues indicate curvature
- Positive definite $\Rightarrow$ local minimum
- Negative definite $\Rightarrow$ local maximum
- Mixed signs $\Rightarrow$ saddle point

### Deep Learning Relevance

- Condition number: $\kappa = \lambda_{max}/\lambda_{min}$
- High condition number $\Rightarrow$ ill-conditioned optimization
- Motivates adaptive learning rates (Adam, etc.)

---

## 2.7 Activation Function Derivatives

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Issue**: Vanishing gradients when $|x|$ large

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

**Advantage**: No vanishing gradient for positive inputs

### Softmax

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Jacobian:
$$\frac{\partial \text{softmax}_i}{\partial z_j} = \text{softmax}_i(\delta_{ij} - \text{softmax}_j)$$

---

## 2.8 Gradient of Common Loss Functions

### Mean Squared Error

$$L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$$

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

### Cross-Entropy (Binary)

$$L = -\frac{1}{n}\sum_i [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)$$

### Cross-Entropy with Softmax

The combined gradient simplifies beautifully:
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

---

## 2.9 Vector Calculus Identities

### Useful Identities

| Expression | Gradient |
|------------|----------|
| $\mathbf{a}^T\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{A}\mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |
| $\|\mathbf{x}\|_2^2$ | $2\mathbf{x}$ |
| $\|\mathbf{Ax} - \mathbf{b}\|_2^2$ | $2\mathbf{A}^T(\mathbf{Ax} - \mathbf{b})$ |

### Matrix Calculus Rules

$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{AX}) = \mathbf{A}^T$$

$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^T\mathbf{AX}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{X}$$

---

## Code Implementation

```python
import numpy as np

# Numerical gradient
def numerical_gradient(f, x, eps=1e-7):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Gradient check
def gradient_check(f, grad_f, x, eps=1e-5):
    numerical = numerical_gradient(f, x, eps)
    analytical = grad_f(x)
    error = np.linalg.norm(numerical - analytical) / (np.linalg.norm(numerical) + np.linalg.norm(analytical))
    return error < 1e-5, error
```

---

## Exercises

### Exercise 2.1 (Section 2.1)
Compute derivatives:
1. $f(x) = x^3 e^{-x}$
2. $f(x) = \ln(1 + e^x)$ (softplus)
3. $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ (tanh)

### Exercise 2.2 (Section 2.3)
For $f(x, y, z) = x^2y + yz^2 + xz$:
1. Compute the gradient $\nabla f$
2. Evaluate at point $(1, 2, 3)$
3. In which direction does $f$ increase most rapidly?

### Exercise 2.3 (Section 2.4)
Using the chain rule, derive the gradient of:
$$L = (y - \sigma(w^T x + b))^2$$
with respect to $w$ and $b$.

### Exercise 2.4 (Section 2.7)
Implement gradient checking for a small neural network. Verify that analytical gradients match numerical gradients.

---

## Summary

- Derivatives measure change and enable optimization
- Gradients point toward steepest ascent
- Chain rule enables backpropagation through layers
- Jacobians and Hessians describe local geometry
- Understanding activation derivatives helps diagnose training issues

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 3.

### Supplementary Reading
- Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 4.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*, Appendix A.

---

## Course Progression

This calculus foundation enables:
- **Week 3**: Gradient-based optimization algorithms (SGD, Adam)
- **Week 4**: Understanding neural network computations
- **Week 5**: Deriving backpropagation from chain rule
- **All subsequent weeks**: Computing and interpreting gradients

