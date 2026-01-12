# Week 12: Kernel Methods & Neural Networks

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapters 12-13: Kernel Methods and Fully Connected Neural Networks**.

---

## Overview

This week covers two powerful approaches for nonlinear learning: kernel methods and neural networks.

---

## Learning Objectives

- Understand the kernel trick and its applications
- Apply kernel SVM for nonlinear classification
- Understand neural network architecture
- Implement forward and backward propagation

---

# Part 1: Kernel Methods (Chapter 12)

## 12.1 Introduction

### The Problem

Want nonlinear models but:
- Feature space may be very high-dimensional
- Computing features explicitly is expensive
- Storage is prohibitive

### The Solution: Kernels

Compute inner products in feature space without explicit features.

---

## 12.2 Fixed-Shape Universal Approximators

Radial Basis Functions (RBFs):
$$f(x) = \sum_{j=1}^{M} w_j \phi_j(x) = \sum_{j=1}^{M} w_j K(x, c_j)$$

Where $K$ is a kernel centered at $c_j$.

---

## 12.3 The Kernel Trick

### Key Observation

Many algorithms only need inner products $\langle \phi(x_i), \phi(x_j) \rangle$.

### The Kernel Function

$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

Computes inner product in feature space without explicit $\phi$.

### Example: Polynomial Kernel

For $K(x, x') = (1 + x^T x')^2$ with $x, x' \in \mathbb{R}^2$:

This implicitly computes:
$$\phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2]^T$$

6-dimensional feature space from 2D input!

---

## 12.4 Kernels as Measures of Similarity

### Common Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| Linear | $x^T x'$ | No transformation |
| Polynomial | $(c + x^T x')^d$ | Polynomial features |
| RBF/Gaussian | $\exp(-\gamma\|x-x'\|^2)$ | Infinite-dimensional |
| Sigmoid | $\tanh(\alpha x^T x' + c)$ | Not always valid |

### RBF Kernel Properties

- Universal approximator
- Localized influence
- Parameter $\gamma$ controls width

---

## 12.5 Optimization of Kernelized Models

### Kernel Ridge Regression

$$\min_\alpha \|y - K\alpha\|^2 + \lambda \alpha^T K \alpha$$

Solution: $\alpha = (K + \lambda I)^{-1} y$

### Kernel SVM

$$\min_\alpha \frac{1}{2}\alpha^T Q \alpha - \sum_i \alpha_i$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i y_i \alpha_i = 0$

---

## Implementation: Kernel Methods

```python
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

# Kernel SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# Kernel Ridge Regression
krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
krr.fit(X_train, y_train)

# Custom kernel
def my_kernel(X, Y):
    return np.dot(X, Y.T)  # Linear kernel

svm_custom = SVC(kernel=my_kernel)
```

---

# Part 2: Neural Networks (Chapter 13)

## 13.1 Introduction

### Why Neural Networks?

- Learn features automatically
- Hierarchical representations
- State-of-the-art on many tasks

---

## 13.2 Fully Connected Neural Networks

### Architecture

$$h^{(0)} = x$$
$$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)}), \quad l = 1, \ldots, L-1$$
$$f(x) = W^{(L)} h^{(L-1)} + b^{(L)}$$

### Parameters

- $W^{(l)}$: Weight matrices
- $b^{(l)}$: Bias vectors
- Total: $\sum_l (n_{l-1} + 1) \times n_l$

### Notation

| Symbol | Meaning |
|--------|---------|
| $L$ | Number of layers |
| $n_l$ | Units in layer $l$ |
| $h^{(l)}$ | Activations at layer $l$ |
| $\sigma$ | Activation function |

---

## 13.3 Activation Functions

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Output: $(0, 1)$
- Problem: Vanishing gradients

### Tanh

$$\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

- Output: $(-1, 1)$
- Zero-centered

### ReLU (Recommended)

$$\sigma(z) = \max(0, z)$$

- No saturation for $z > 0$
- Sparse activations
- Problem: Dying neurons

### Leaky ReLU

$$\sigma(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

- Prevents dying neurons

---

## 13.4 The Backpropagation Algorithm

### Forward Pass

```python
def forward(x, weights, biases):
    h = x
    activations = [h]
    
    for W, b in zip(weights[:-1], biases[:-1]):
        z = W @ h + b
        h = relu(z)
        activations.append(h)
    
    # Output layer (no activation for regression)
    y = weights[-1] @ h + biases[-1]
    return y, activations
```

### Backward Pass

Using chain rule:
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

Where $\delta^{(l)}$ is the error signal at layer $l$.

```python
def backward(y, y_true, activations, weights):
    # Output error
    delta = 2 * (y - y_true)
    gradients = []
    
    for l in reversed(range(len(weights))):
        grad_W = np.outer(delta, activations[l])
        gradients.append(grad_W)
        
        if l > 0:
            delta = (weights[l].T @ delta) * relu_derivative(activations[l])
    
    return gradients[::-1]
```

---

## 13.5 Optimization of Neural Network Models

### Challenges

- Non-convex loss landscape
- Many local minima and saddle points
- Careful initialization required

### Modern Optimizers

| Optimizer | Key Idea |
|-----------|----------|
| SGD + Momentum | Accumulated velocity |
| Adam | Adaptive learning rates |
| RMSprop | Running average of gradients |

---

## 13.6 Batch Normalization

### Algorithm

1. Compute mini-batch mean and variance
2. Normalize: $\hat{h} = (h - \mu) / \sqrt{\sigma^2 + \epsilon}$
3. Scale and shift: $\tilde{h} = \gamma \hat{h} + \beta$

### Benefits

- Faster training
- Higher learning rates
- Regularization effect

---

## 13.7 Cross-Validation via Early Stopping

### The Idea

Stop training when validation error starts increasing.

### Implementation

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_data)
    val_loss = evaluate(model, val_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

---

## Implementation: Neural Network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create model
model = MLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)

# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
```

---

## Exercises

### Exercise 12.1 (Chapter 12)
Compare linear vs RBF kernel SVM on a non-linearly separable dataset.

### Exercise 13.1 (Chapter 13)
Implement a 2-layer neural network from scratch using NumPy.

### Exercise 13.2 (Chapter 13)
Compare different activation functions on a regression problem.

---

## Summary

- Kernel trick: Implicit high-dimensional features
- RBF kernel: Most common, universal approximator
- Neural networks: Learned feature hierarchies
- Backpropagation: Efficient gradient computation
- Modern techniques: BatchNorm, Adam, early stopping

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapters 12-13**.

### Supplementary Reading
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 5-6.
- Scholkopf, B., & Smola, A. J. (2002). *Learning with Kernels*.

---

## Next Week Preview

**Week 13: Tree-Based Learners & Advanced Topics** (Chapter 14)
- Decision trees
- Gradient boosting
- Random forests
