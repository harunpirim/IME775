# Week 6: Regularization and Generalization in Deep Learning

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 7: Regularization Techniques**.

---

## Overview

This module covers regularization techniques that prevent overfitting and improve generalization in deep neural networks.

---

## Learning Objectives

- Understand the bias-variance tradeoff in neural networks
- Master L1, L2, and dropout regularization
- Learn batch normalization and its variants
- Implement early stopping and data augmentation

---

## 6.1 The Overfitting Problem

### Bias-Variance Decomposition

Expected error = Bias² + Variance + Irreducible Noise

| Component | Description | High When |
|-----------|-------------|-----------|
| Bias | Model can't capture true pattern | Model too simple |
| Variance | Model changes a lot with different data | Model too complex |
| Noise | Inherent data randomness | Always present |

### Overfitting in Neural Networks

Neural networks are highly expressive (low bias) but can memorize training data (high variance).

**Signs of overfitting**:
- Training loss keeps decreasing
- Validation loss starts increasing
- Gap between train/val accuracy grows

---

## 6.2 Weight Regularization

### L2 Regularization (Weight Decay)

Add penalty on squared weights to loss:

$$L_{total} = L_{data} + \frac{\lambda}{2} \sum_{l} \|W^{(l)}\|_F^2$$

**Gradient update**:
$$W_{t+1} = W_t - \alpha(\nabla L_{data} + \lambda W_t) = (1 - \alpha\lambda)W_t - \alpha\nabla L_{data}$$

**Effect**: Shrinks weights toward zero, prevents large weights.

### L1 Regularization (Sparsity)

$$L_{total} = L_{data} + \lambda \sum_{l} \|W^{(l)}\|_1$$

**Effect**: Encourages sparse weights (many exactly zero).

### Comparison

| Aspect | L2 (Ridge) | L1 (Lasso) |
|--------|------------|------------|
| Effect on weights | Shrinks all | Sparse (some zero) |
| Solution | Closed form exists | No closed form |
| Feature selection | No | Yes (implicit) |
| Typical use | Default regularization | When sparsity needed |

### ElasticNet

Combination of L1 and L2:
$$L_{total} = L_{data} + \lambda_1 \|W\|_1 + \lambda_2 \|W\|_2^2$$

---

## 6.3 Dropout

### Mechanism

During training, randomly set fraction $p$ of neurons to zero:

$$h^{(l)} = \text{mask} \odot \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$$

Where mask is binary with $P(\text{mask}_i = 0) = p$.

### Inverted Dropout

Scale remaining activations by $\frac{1}{1-p}$ during training:
- Training: $h = \frac{1}{1-p} \cdot \text{mask} \odot \sigma(z)$
- Inference: $h = \sigma(z)$ (no dropout)

### Interpretation

1. **Ensemble view**: Training many sub-networks, averaging at test
2. **Regularization view**: Adding noise prevents co-adaptation
3. **Bayesian view**: Approximate variational inference

### Practical Guidelines

| Network Part | Dropout Rate |
|--------------|--------------|
| Input layer | 0.1-0.2 |
| Hidden layers | 0.3-0.5 |
| Before output | 0.0-0.3 |

---

## 6.4 Batch Normalization

### Motivation

Internal covariate shift: Distribution of layer inputs changes during training.

### Mechanism

For mini-batch $B = \{x_1, \ldots, x_m\}$:

**Normalize**:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Where:
$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

**Scale and shift** (learnable):
$$y_i = \gamma \hat{x}_i + \beta$$

### Training vs Inference

| Phase | Mean/Variance Source |
|-------|---------------------|
| Training | Mini-batch statistics |
| Inference | Running averages (exponential moving average) |

### Benefits

1. **Faster training**: Allows higher learning rates
2. **Regularization**: Adds noise via batch statistics
3. **Reduces sensitivity**: To initialization and hyperparameters

### Placement

Two common positions:
- **Pre-activation**: BN → ReLU (original paper)
- **Post-activation**: ReLU → BN

---

## 6.5 Layer Normalization

### Motivation

BatchNorm depends on batch size and doesn't work well for:
- Small batches
- Recurrent networks
- Variable-length sequences

### Mechanism

Normalize across features for each sample:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Where $\mu, \sigma$ computed over feature dimension (not batch).

### Comparison

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalize over | Batch | Features |
| Batch size dependency | Yes | No |
| Good for | CNNs | Transformers, RNNs |
| At inference | Uses running stats | Same as training |

---

## 6.6 Early Stopping

### Concept

Stop training when validation loss starts increasing:

1. Monitor validation loss each epoch
2. Save model with best validation loss
3. Stop if no improvement for $k$ epochs (patience)

### Implementation

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### Regularization View

Early stopping is equivalent to L2 regularization (approximately):
- Shorter training → smaller effective weights
- Like constraining weight norm

---

## 6.7 Data Augmentation

### Concept

Artificially expand training data by applying transformations.

### Image Augmentations

| Transformation | Example |
|----------------|---------|
| Geometric | Rotation, flip, crop, scale |
| Color | Brightness, contrast, saturation |
| Noise | Gaussian noise, blur |
| Erasing | Random cutout, CutMix |
| Mixing | Mixup, CutMix |

### Mixup

Create virtual training examples:
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

### Text/Sequence Augmentations

| Transformation | Example |
|----------------|---------|
| Synonym replacement | "happy" → "joyful" |
| Random insertion | Insert random word |
| Random swap | Swap two words |
| Back-translation | English → French → English |

---

## 6.8 Other Regularization Techniques

### Label Smoothing

Instead of hard labels [0, 0, 1, 0]:
$$y_{smooth} = (1 - \epsilon)y_{hard} + \frac{\epsilon}{K}$$

For K classes. Prevents overconfident predictions.

### Gradient Clipping

Prevent exploding gradients:
$$g \leftarrow \min\left(1, \frac{\theta}{\|g\|}\right) g$$

### Spectral Normalization

Constrain Lipschitz constant of each layer:
$$\bar{W} = \frac{W}{\sigma(W)}$$

Where $\sigma(W)$ is largest singular value.

---

## 6.9 Choosing Regularization

### General Guidelines

```
Start with:
├── Baseline (no regularization)
├── + Dropout (0.3-0.5)
├── + Weight decay (1e-4)
├── + BatchNorm / LayerNorm
├── + Data augmentation
└── + Early stopping
```

### By Problem Type

| Task | Recommended |
|------|-------------|
| Image classification | Dropout, augmentation, weight decay |
| NLP/Transformers | LayerNorm, dropout, label smoothing |
| Small datasets | Strong augmentation, dropout, early stopping |
| Large datasets | Less regularization needed, BatchNorm |

---

## Code Implementation

```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x
    
    def backward(self, grad_output):
        return grad_output * self.mask

class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True
    
    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta

def l2_regularization(weights, lambda_reg):
    """Compute L2 regularization loss and gradient."""
    reg_loss = 0.5 * lambda_reg * sum(np.sum(W**2) for W in weights)
    reg_grads = [lambda_reg * W for W in weights]
    return reg_loss, reg_grads
```

---

## Exercises

### Exercise 6.1 (Section 6.2)
Train a network on a small dataset with and without L2 regularization. Compare training and validation curves.

### Exercise 6.2 (Section 6.3)
Implement dropout from scratch and verify it produces different outputs each forward pass during training.

### Exercise 6.3 (Section 6.4)
Add BatchNorm to a deep network. Compare training speed with and without it.

### Exercise 6.4 (Section 6.7)
Apply Mixup to an image classification task and measure improvement.

---

## Summary

- Regularization prevents overfitting by constraining model complexity
- L2/L1 penalties shrink weights
- Dropout creates implicit ensemble
- Batch/Layer Norm stabilize training and regularize
- Early stopping is a simple but effective technique
- Data augmentation expands effective dataset size

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 7.

### Supplementary Reading
- Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting."
- Ioffe, S. & Szegedy, C. (2015). "Batch Normalization."

---

## Course Progression

Regularization techniques apply throughout:
- **Week 7**: Batch normalization in modern architectures
- **Week 9**: Dropout in CNN fully-connected layers
- **Week 11**: Dropout in RNN/LSTM models
- **Week 13**: Layer normalization in Transformers
- **All projects**: Essential for achieving good generalization

