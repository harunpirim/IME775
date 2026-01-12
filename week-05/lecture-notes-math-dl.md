# Week 5: Backpropagation - The Engine of Deep Learning

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 6: Backpropagation and Training**.

---

## Overview

This module covers backpropagation, the fundamental algorithm that enables training deep neural networks by efficiently computing gradients.

---

## Learning Objectives

- Understand backpropagation as reverse-mode automatic differentiation
- Derive gradients for common layers
- Implement backpropagation from scratch
- Connect computational graphs to gradient computation

---

## 5.1 The Gradient Problem

### Why Gradients?

Training a neural network requires minimizing a loss function:
$$\theta^* = \arg\min_\theta L(\theta)$$

Using gradient descent:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L$$

**Challenge**: Networks have millions of parameters - how do we compute gradients efficiently?

### Naive Approach: Numerical Gradients

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon e_i) - L(\theta - \epsilon e_i)}{2\epsilon}$$

**Problem**: For $n$ parameters, need $2n$ forward passes!

### Solution: Backpropagation

Compute **all** gradients in one backward pass.

---

## 5.2 Computational Graphs

### Representing Computation

Any neural network can be represented as a **directed acyclic graph (DAG)**:
- Nodes: Operations or variables
- Edges: Data flow

### Example: Simple Network

For $L = (y - \sigma(wx + b))^2$:

```
x → [*w] → [+b] → [σ] → [-y] → [²] → L
```

### Forward Pass

Compute outputs by traversing graph from inputs to outputs.

### Backward Pass

Compute gradients by traversing graph from outputs to inputs.

---

## 5.3 The Chain Rule Revisited

### Single Path

For $L = f(g(h(x)))$:
$$\frac{\partial L}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

### Multiple Paths

If $x$ affects $L$ through multiple paths:
$$\frac{\partial L}{\partial x} = \sum_{\text{all paths}} \prod_{\text{edges}} \frac{\partial}{\partial}$$

### Local Gradients

Each node computes:
1. **Forward**: Output given inputs
2. **Backward**: $\frac{\partial L}{\partial \text{input}} = \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{input}}$

---

## 5.4 Backpropagation Algorithm

### Overview

1. **Forward pass**: Compute all activations, store intermediate values
2. **Compute loss**: Evaluate $L$ at output
3. **Backward pass**: Propagate gradients from output to inputs

### Detailed Steps

**Forward pass**:
```
for l = 1 to L:
    z[l] = W[l] @ h[l-1] + b[l]
    h[l] = activation(z[l])
```

**Backward pass**:
```
delta[L] = dL/dh[L] * activation'(z[L])

for l = L-1 to 1:
    dW[l] = h[l-1].T @ delta[l+1]
    db[l] = sum(delta[l+1])
    delta[l] = delta[l+1] @ W[l+1].T * activation'(z[l])
```

### Key Insight

Gradient at layer $l$ depends only on:
1. Gradient from layer $l+1$ (upstream gradient)
2. Local gradient of layer $l$'s operation

---

## 5.5 Gradient Derivations

### Linear Layer: $z = Wx + b$

**Forward**: $z = Wx + b$

**Backward**:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^T$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$$
$$\frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial z}$$

### ReLU: $h = \max(0, z)$

**Forward**: $h = \max(0, z)$

**Backward**:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot \mathbb{1}[z > 0]$$

### Sigmoid: $h = \sigma(z) = \frac{1}{1 + e^{-z}}$

**Forward**: $h = \sigma(z)$

**Backward**:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot h(1 - h)$$

### Softmax with Cross-Entropy

**Combined gradient** (elegant simplification):
$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

Where $p = \text{softmax}(z)$ and $y$ is one-hot target.

### Mean Squared Error: $L = \frac{1}{n}\sum(y - \hat{y})^2$

**Backward**:
$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$

---

## 5.6 Matrix Calculus for Batches

### Batch Notation

For batch of $B$ samples:
- $X \in \mathbb{R}^{B \times n}$: Input batch
- $W \in \mathbb{R}^{n \times m}$: Weights
- $Z = XW \in \mathbb{R}^{B \times m}$: Output batch

### Batch Gradients

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Z}$$

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W^T$$

$$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial Z}$$

---

## 5.7 Implementation

### Complete Backprop Implementation

```python
import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, x):
        self.x = x  # Cache for backward
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

class ReLU(Layer):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask

class Sigmoid(Layer):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class MSELoss:
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)
    
    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params.extend([layer.W, layer.b])
                grads.extend([layer.grad_W, layer.grad_b])
        return params, grads
```

### Training Loop

```python
# Initialize
model = NeuralNetwork([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])
loss_fn = MSELoss()

# Training
for epoch in range(epochs):
    # Forward
    output = model.forward(X_batch)
    loss = loss_fn.forward(output, y_batch)
    
    # Backward
    grad = loss_fn.backward()
    model.backward(grad)
    
    # Update
    params, grads = model.get_params_and_grads()
    for p, g in zip(params, grads):
        p -= learning_rate * g
```

---

## 5.8 Gradient Checking

### Numerical Verification

Always verify analytical gradients against numerical gradients:

```python
def gradient_check(model, loss_fn, X, y, eps=1e-5):
    # Analytical gradient
    output = model.forward(X)
    loss = loss_fn.forward(output, y)
    grad = loss_fn.backward()
    model.backward(grad)
    
    params, analytical_grads = model.get_params_and_grads()
    
    # Numerical gradient
    for p, a_grad in zip(params, analytical_grads):
        numerical_grad = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old_val = p[idx]
            
            p[idx] = old_val + eps
            loss_plus = loss_fn.forward(model.forward(X), y)
            
            p[idx] = old_val - eps
            loss_minus = loss_fn.forward(model.forward(X), y)
            
            p[idx] = old_val
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare
        diff = np.linalg.norm(a_grad - numerical_grad) / (np.linalg.norm(a_grad) + np.linalg.norm(numerical_grad) + 1e-8)
        print(f"Relative difference: {diff:.2e}")
```

### Typical Threshold

- Relative error < $10^{-5}$: Correct
- Relative error $10^{-5}$ to $10^{-3}$: Check carefully
- Relative error > $10^{-3}$: Bug likely

---

## 5.9 Vanishing and Exploding Gradients

### The Problem

In deep networks, gradients can become very small or very large:

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \prod_{l=1}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}}$$

If each factor is < 1: Vanishing
If each factor is > 1: Exploding

### Solutions

| Problem | Solution |
|---------|----------|
| Vanishing | ReLU activation, skip connections, proper initialization |
| Exploding | Gradient clipping, proper initialization |
| Both | Batch normalization, layer normalization |

---

## Exercises

### Exercise 5.1 (Section 5.3)
Draw the computational graph for $L = (y - \text{ReLU}(w_2 \cdot \text{ReLU}(w_1 x + b_1) + b_2))^2$.

### Exercise 5.2 (Section 5.5)
Derive the backward pass for the tanh activation function.

### Exercise 5.3 (Section 5.7)
Implement backpropagation for a 3-layer network from scratch and verify with gradient checking.

### Exercise 5.4 (Section 5.9)
Create a deep network (10+ layers) with sigmoid activations and observe vanishing gradients. Compare with ReLU.

---

## Summary

- Backpropagation efficiently computes gradients via chain rule
- Each layer computes local gradients and passes upstream gradient
- Computational graphs make gradient flow explicit
- Gradient checking is essential for debugging
- Proper initialization and activation functions prevent gradient issues

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 6.

### Supplementary Reading
- Rumelhart, Hinton & Williams (1986). "Learning representations by back-propagating errors."
- Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 6.

---

## Course Progression

Backpropagation is the training engine for:
- **Week 6**: Understanding how regularization affects gradients
- **Week 7**: Gradient flow through skip connections (ResNets)
- **Week 9**: Backprop through convolutional layers
- **Week 11**: Backprop through time in RNNs
- **Week 13**: Attention gradient computations

