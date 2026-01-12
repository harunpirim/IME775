# Week 4: Neural Network Foundations - The Perceptron to MLPs

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 5: Neural Networks Basics**.

---

## Overview

This module introduces the architecture and mathematics of neural networks, from single neurons to multi-layer perceptrons (MLPs).

---

## Learning Objectives

- Understand the perceptron as a linear classifier
- Master the mathematics of multi-layer networks
- Learn forward propagation computation
- Understand universal approximation theorem

---

## 4.1 The Perceptron

### Biological Inspiration

A neuron:
1. Receives inputs (dendrites)
2. Processes them (cell body)
3. Produces output (axon)

### Mathematical Model

$$y = \sigma(w^T x + b) = \sigma\left(\sum_{i=1}^n w_i x_i + b\right)$$

Where:
- $x \in \mathbb{R}^n$: Input features
- $w \in \mathbb{R}^n$: Weights
- $b \in \mathbb{R}$: Bias
- $\sigma$: Activation function
- $y$: Output

### Linear Classification View

The perceptron defines a **hyperplane**:
$$w^T x + b = 0$$

- Points where $w^T x + b > 0$ → Class 1
- Points where $w^T x + b < 0$ → Class 0

### Limitations

**XOR Problem**: Cannot classify linearly non-separable data.

| x₁ | x₂ | XOR |
|----|----|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Solution**: Stack multiple layers!

---

## 4.2 Activation Functions

### Purpose

- Introduce **non-linearity**
- Without activation: composition of linear functions = linear function
- Enable learning complex patterns

### Common Activations

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0, 1) | Smooth, saturates |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Zero-centered |
| ReLU | $\max(0, x)$ | [0, ∞) | Non-saturating, sparse |
| Leaky ReLU | $\max(\alpha x, x)$ | (-∞, ∞) | No dead neurons |
| GELU | $x \cdot \Phi(x)$ | (-∞, ∞) | Smooth ReLU |
| Swish | $x \cdot \sigma(x)$ | (-∞, ∞) | Self-gated |

### Modern Recommendations

- **Hidden layers**: ReLU, GELU, or Swish
- **Output (classification)**: Softmax
- **Output (regression)**: Linear (no activation)

---

## 4.3 Multi-Layer Perceptron (MLP)

### Architecture

```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output
  x    →    h₁         →    h₂          → ... →    y
```

### Forward Propagation

For an L-layer network:

$$h^{(0)} = x$$
$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
$$h^{(l)} = \sigma^{(l)}(z^{(l)})$$
$$y = h^{(L)}$$

### Matrix Formulation

For batch of inputs $X \in \mathbb{R}^{B \times n}$:

$$H^{(l)} = \sigma(H^{(l-1)} W^{(l)T} + \mathbf{1}b^{(l)T})$$

### Dimension Tracking

| Layer | Input | Weights | Bias | Output |
|-------|-------|---------|------|--------|
| 1 | (B, n₀) | (n₁, n₀) | (n₁,) | (B, n₁) |
| 2 | (B, n₁) | (n₂, n₁) | (n₂,) | (B, n₂) |
| ... | ... | ... | ... | ... |
| L | (B, n_{L-1}) | (n_L, n_{L-1}) | (n_L,) | (B, n_L) |

---

## 4.4 Universal Approximation Theorem

### Statement

A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$, given appropriate activation functions.

### Intuition

- Each hidden neuron defines a "bump" or "ridge"
- Combining many neurons can approximate any shape
- More neurons → finer approximation

### Practical Implications

| Aspect | Single Wide Layer | Multiple Deep Layers |
|--------|-------------------|---------------------|
| Expressiveness | Sufficient (theory) | More efficient |
| Parameters | Many | Fewer for same capacity |
| Learning | Hard to optimize | Hierarchical features |
| Practice | Rarely used | Standard approach |

**Key insight**: Depth provides exponential expressiveness gains.

---

## 4.5 Loss Functions for Neural Networks

### Regression

**Mean Squared Error (MSE)**:
$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Mean Absolute Error (MAE)**:
$$L = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

### Binary Classification

**Binary Cross-Entropy**:
$$L = -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Multi-Class Classification

**Categorical Cross-Entropy** (with softmax):
$$L = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})$$

Where $y_{ic}$ is one-hot encoded.

---

## 4.6 Network Architecture Design

### Width vs Depth

| Aspect | Wider | Deeper |
|--------|-------|--------|
| Expressiveness | Linear in width | Exponential in depth |
| Training | Easier | Harder (gradients) |
| Compute | Parallelizable | Sequential |
| Features | Single level | Hierarchical |

### Common Architectures

**Regression**:
```
Input(n) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
```

**Binary Classification**:
```
Input(n) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

**Multi-Class (C classes)**:
```
Input(n) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(C, Softmax)
```

### Hyperparameters

| Parameter | Typical Range | How to Choose |
|-----------|---------------|---------------|
| Hidden layers | 2-5 | Start small, increase if underfitting |
| Hidden units | 32-512 | Power of 2, depends on data |
| Learning rate | 1e-4 to 1e-2 | Grid search, schedulers |
| Batch size | 32-256 | Memory constraints, stability |

---

## 4.7 Vectorized Implementation

### Efficient Forward Pass

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X):
        self.activations = [X]
        self.pre_activations = []
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.pre_activations.append(z)
            
            if i < len(self.weights) - 1:
                h = relu(z)  # Hidden layers
            else:
                h = z  # Output layer (linear for regression, softmax for classification)
            
            self.activations.append(h)
        
        return self.activations[-1]
```

### Batch Processing Advantage

Single sample: $O(n)$ matrix-vector multiplications
Batch of B samples: $O(n)$ matrix-matrix multiplications (same cost, B× throughput)

---

## Code Example: Full MLP Implementation

```python
import numpy as np

class SimpleMLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initialize MLP with given architecture.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of outputs
        """
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_dims) - 1):
            W = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros(layer_dims[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward pass through the network."""
        self.cache = {'activations': [X], 'pre_activations': []}
        
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            self.cache['pre_activations'].append(z)
            
            if i < len(self.weights) - 1:
                h = np.maximum(0, z)  # ReLU for hidden
            else:
                h = z  # Linear output
            
            self.cache['activations'].append(h)
        
        return h
    
    def predict(self, X):
        """Make predictions."""
        return self.forward(X)
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

# Example usage
mlp = SimpleMLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
print(f"Total parameters: {mlp.count_parameters():,}")

# Forward pass
X = np.random.randn(32, 784)  # Batch of 32 samples
output = mlp.predict(X)
print(f"Output shape: {output.shape}")
```

---

## Exercises

### Exercise 4.1 (Section 4.1)
Draw the decision boundary for a single neuron with weights $w = [1, 2]$ and bias $b = -1$.

### Exercise 4.2 (Section 4.3)
For a network with layers [784, 256, 128, 10]:
1. How many total parameters?
2. What are the dimensions of each weight matrix?

### Exercise 4.3 (Section 4.4)
Implement a 2-layer network that learns the XOR function. Visualize the learned decision boundary.

### Exercise 4.4 (Section 4.7)
Compare the time to process 1000 samples one-at-a-time vs as a batch of 1000.

---

## Summary

- Single neurons compute weighted sums with nonlinear activation
- Stacking layers enables learning complex non-linear functions
- Universal approximation: one hidden layer suffices (in theory)
- Deep networks learn hierarchical representations efficiently
- Vectorized operations enable efficient batch processing

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 5.

### Supplementary Reading
- Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 6.
- Nielsen, M. (2015). *Neural Networks and Deep Learning*, Chapter 1-2.

---

## Course Progression

This neural network foundation prepares you for:
- **Week 5**: Understanding how backpropagation trains these networks
- **Week 6**: Regularization techniques for MLPs
- **Week 7**: Modern architectures that extend basic MLPs
- **Weeks 9-13**: CNNs, RNNs, Transformers all build on MLP concepts

