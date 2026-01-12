# Week 3: Gradient-Based Optimization for Deep Learning

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 4: Optimization Algorithms**.

---

## Overview

This module covers gradient-based optimization algorithms used to train neural networks, from basic gradient descent to modern adaptive methods.

---

## Learning Objectives

- Understand gradient descent and its variants
- Master momentum-based acceleration
- Learn adaptive learning rate methods (AdaGrad, RMSprop, Adam)
- Connect optimization to neural network training dynamics

---

## 3.1 Gradient Descent: The Foundation

### Basic Update Rule

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)$$

Where:
- $\theta$: Model parameters
- $\alpha$: Learning rate (step size)
- $\nabla_\theta L$: Gradient of loss with respect to parameters

### Intuition

- Gradient points toward steepest ascent
- Negative gradient points toward steepest descent
- Take steps proportional to gradient magnitude

### Challenges

| Challenge | Description |
|-----------|-------------|
| Learning rate selection | Too large → diverge, too small → slow |
| Local minima | Can get stuck (less issue in high dimensions) |
| Saddle points | Gradients vanish but not at minimum |
| Ill-conditioning | Different curvature in different directions |

---

## 3.2 Stochastic Gradient Descent (SGD)

### Mini-Batch Gradient

Instead of full gradient over all data:

$$\nabla_\theta L \approx \frac{1}{|B|} \sum_{i \in B} \nabla_\theta L_i(\theta)$$

Where $B$ is a random mini-batch.

### SGD Update

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{|B|} \sum_{i \in B} \nabla_\theta L_i(\theta_t)$$

### Properties

| Aspect | Full Batch | Mini-Batch SGD |
|--------|------------|----------------|
| Gradient accuracy | Exact | Noisy estimate |
| Memory | O(N) | O(batch_size) |
| Updates per epoch | 1 | N/batch_size |
| Can escape local minima | Hard | Easier (noise helps) |

### Typical Batch Sizes

- 32, 64, 128, 256 common choices
- Larger batches → more stable but need larger learning rate
- Smaller batches → more noise, can help generalization

---

## 3.3 Momentum

### Problem with Vanilla SGD

- Oscillates in steep dimensions
- Slow progress in flat dimensions
- Especially bad when curvature varies greatly

### Momentum Update

Accumulate velocity in consistent gradient directions:

$$v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

Where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

### Intuition

Like a ball rolling down a hill:
- Accelerates in consistent direction
- Dampens oscillations in inconsistent directions

### Nesterov Accelerated Gradient (NAG)

Look-ahead momentum:

$$v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t - \alpha \beta v_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

**Intuition**: Evaluate gradient at the anticipated next position.

---

## 3.4 Adaptive Learning Rates

### The Problem

Different parameters may need different learning rates:
- Features with large gradients → smaller steps
- Features with small gradients → larger steps

### AdaGrad

Adapt learning rate based on historical gradient magnitudes:

$$g_{t+1} = g_t + (\nabla_\theta L)^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{g_{t+1} + \epsilon}} \nabla_\theta L$$

**Problem**: $g_t$ only grows → learning rate goes to zero.

### RMSprop

Exponential moving average instead of accumulation:

$$g_{t+1} = \beta g_t + (1-\beta)(\nabla_\theta L)^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{g_{t+1} + \epsilon}} \nabla_\theta L$$

**Typical**: $\beta = 0.9$

### Adam (Adaptive Moment Estimation)

Combines momentum with adaptive learning rates:

**First moment** (momentum):
$$m_{t+1} = \beta_1 m_t + (1-\beta_1) \nabla_\theta L$$

**Second moment** (adaptive scaling):
$$v_{t+1} = \beta_2 v_t + (1-\beta_2) (\nabla_\theta L)^2$$

**Bias correction**:
$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}$$
$$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$

**Update**:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## 3.5 Learning Rate Schedules

### Step Decay

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Reduce by factor $\gamma$ every $s$ epochs.

### Exponential Decay

$$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$$

### Cosine Annealing

$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T}\pi))$$

### Warmup

Start with small learning rate, gradually increase:
$$\alpha_t = \frac{t}{T_{warmup}} \alpha_{target}$$ for $t < T_{warmup}$

Important for large batch training and transformers.

### One-Cycle Policy

- Warmup → max learning rate → anneal down
- Often achieves better results in fewer epochs

---

## 3.6 Comparison of Optimizers

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD | Simple, generalizes well | Slow, needs tuning | Final training |
| SGD+Momentum | Faster than SGD | Extra hyperparameter | Standard choice |
| Adam | Fast, adaptive | Can generalize worse | Quick prototyping |
| AdamW | Better generalization | More hyperparameters | Transformers |

### Practical Guidelines

1. **Start with Adam** for quick experiments
2. **Switch to SGD+Momentum** for final training if time allows
3. **Use learning rate warmup** for large models
4. **Tune learning rate first**, then other hyperparameters

---

## 3.7 Weight Initialization

### Why It Matters

- Poor initialization → vanishing/exploding gradients
- Can prevent learning entirely

### Xavier/Glorot Initialization

For layer with $n_{in}$ inputs and $n_{out}$ outputs:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

**Best for**: Sigmoid, Tanh activations

### He Initialization

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**Best for**: ReLU and variants

### Implementation

```python
import torch.nn as nn

# Xavier
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# He (Kaiming)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

---

## 3.8 Gradient Clipping

### Problem: Exploding Gradients

Gradients can become very large, destabilizing training.

### Gradient Norm Clipping

If $\|\nabla_\theta L\| > \text{threshold}$:
$$\nabla_\theta L \leftarrow \frac{\text{threshold}}{\|\nabla_\theta L\|} \nabla_\theta L$$

### Implementation

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# TensorFlow
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

### When to Use

- RNNs/LSTMs (prone to exploding gradients)
- Very deep networks
- When loss suddenly spikes during training

---

## Code Implementation

```python
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None
    
    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] + g
            p -= self.lr * self.velocities[i]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## Exercises

### Exercise 3.1 (Section 3.1-3.2)
Compare full-batch GD vs SGD on a simple quadratic function. Plot convergence curves.

### Exercise 3.2 (Section 3.3)
Visualize the effect of momentum on a 2D function with different curvatures in x and y directions.

### Exercise 3.3 (Section 3.4)
Implement Adam from scratch and verify against PyTorch's implementation.

### Exercise 3.4 (Section 3.5)
Train a small CNN with different learning rate schedules. Compare final accuracy and training dynamics.

---

## Summary

- SGD with mini-batches enables training on large datasets
- Momentum accelerates convergence in consistent directions
- Adaptive methods (Adam) adjust per-parameter learning rates
- Learning rate schedules improve final performance
- Proper initialization prevents gradient issues

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 4.

### Supplementary Reading
- Ruder, S. (2016). "An overview of gradient descent optimization algorithms."
- Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 8.

---

## Course Progression

These optimization techniques are essential for:
- **Week 4**: Training neural networks effectively
- **Week 5**: Understanding backpropagation's role in optimization
- **Week 6**: Regularization interacts with optimizer choice
- **Weeks 7-13**: All architectures trained using these methods

