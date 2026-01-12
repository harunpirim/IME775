# Week 7: Deep Architectures - Modern Building Blocks

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 8: Modern Architectures**.

---

## Overview

This module covers modern architectural components that enable training very deep networks, including skip connections, residual networks, and advanced normalization techniques.

---

## Learning Objectives

- Understand the degradation problem in deep networks
- Master residual connections and their mathematics
- Learn dense connections and efficient architectures
- Understand squeeze-and-excitation and attention mechanisms

---

## 7.1 The Degradation Problem

### Deep Networks Don't Always Work

Paradox: Adding more layers can **hurt** performance, even on training data.

| Network Depth | Training Error | Observation |
|---------------|----------------|-------------|
| 20 layers | 4.5% | Good |
| 56 layers | 6.0% | Worse! |

### Not Overfitting

This isn't overfitting (training error increases). It's an **optimization problem**.

### The Identity Mapping Insight

A deeper network should be able to learn at least an identity mapping:
$$F(x) = x$$

But standard networks struggle to learn this!

---

## 7.2 Residual Networks (ResNet)

### Core Idea

Instead of learning $H(x)$, learn the **residual**:
$$F(x) = H(x) - x$$

So:
$$H(x) = F(x) + x$$

### Residual Block

```
     x
     |
     ↓
  [Conv-BN-ReLU]
     |
     ↓
  [Conv-BN]
     |
     + ← x (skip connection)
     |
     ↓
   [ReLU]
     |
     ↓
   output
```

### Mathematical Formulation

$$y = F(x, \{W_i\}) + x$$

Where $F$ is the residual function (typically 2-3 conv layers).

### Why It Works

1. **Easy to learn identity**: If optimal $H(x) = x$, just set $F(x) = 0$
2. **Gradient flow**: Skip connection provides direct gradient path
3. **Ensemble view**: Unraveled view shows exponential path combinations

### Gradient Flow Analysis

Without skip connections:
$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i}$$

With skip connections:
$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F_i\right)$$

The "1" ensures gradients flow directly!

---

## 7.3 ResNet Variants

### Basic Block (ResNet-18, 34)

Two 3×3 convolutions:
```
x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU
```

### Bottleneck Block (ResNet-50, 101, 152)

1×1 → 3×3 → 1×1 convolutions (reduces computation):
```
x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU
```

### Pre-Activation ResNet

Move BN and ReLU before convolution:
```
x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
```

**Benefits**: Better gradient flow, improved performance.

### ResNeXt

Parallel paths with grouped convolutions:
$$y = x + \sum_{i=1}^{C} T_i(x)$$

Increases "cardinality" (number of paths) instead of depth/width.

---

## 7.4 Dense Networks (DenseNet)

### Core Idea

Connect **every** layer to every subsequent layer:
$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

Where $[\cdot]$ denotes concatenation.

### Advantages

1. **Feature reuse**: All previous features available
2. **Parameter efficiency**: Fewer parameters needed
3. **Deep supervision**: Every layer has direct access to loss gradient

### Dense Block

```
x₀ ────────────────────────┬───┬───┬──→
   │                       │   │   │
   ↓                       │   │   │
  H₁ ──────────────────┬───┤   │   │
   │                   │   │   │   │
   ↓                   │   │   │   │
  H₂ ──────────────┬───┤   │   │   │
   │               │   │   │   │   │
   ↓               │   │   │   │   │
  H₃ ──────────┬───┤   │   │   │   │
               │   │   │   │   │   │
               ↓   ↓   ↓   ↓   ↓   ↓
            [x₀, x₁, x₂, x₃] → H₄
```

### Growth Rate

Each layer adds $k$ features (growth rate). After $l$ layers:
$$\text{features} = k_0 + l \times k$$

### Transition Layer

Between dense blocks, reduce feature maps:
- 1×1 convolution (compression)
- 2×2 average pooling

---

## 7.5 Squeeze-and-Excitation (SE) Networks

### Channel Attention

Not all channels are equally important. SE blocks learn channel weights.

### Mechanism

1. **Squeeze**: Global average pooling → $\mathbb{R}^{C}$
2. **Excitation**: FC → ReLU → FC → Sigmoid → $\mathbb{R}^{C}$
3. **Scale**: Multiply original features by channel weights

### Mathematical Form

$$\tilde{x}_c = s_c \cdot x_c$$

Where:
$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x)))$$

### SE Block

```
     x ∈ R^(H×W×C)
           |
           ↓
    Global Avg Pool
           |
           ↓
        z ∈ R^C
           |
           ↓
      FC (C → C/r)
           |
           ↓
         ReLU
           |
           ↓
      FC (C/r → C)
           |
           ↓
        Sigmoid
           |
           ↓
        s ∈ R^C
           |
           ↓
       x * s (scale)
           |
           ↓
    output ∈ R^(H×W×C)
```

### Integration

SE blocks can be added to any architecture (ResNet-SE, etc.).

---

## 7.6 Efficient Architectures

### MobileNet: Depthwise Separable Convolutions

Standard convolution: $K \times K \times C_{in} \times C_{out}$ parameters

Depthwise separable:
1. **Depthwise**: $K \times K \times 1 \times C_{in}$ (spatial filtering per channel)
2. **Pointwise**: $1 \times 1 \times C_{in} \times C_{out}$ (channel mixing)

**Reduction**: ~$\frac{1}{K^2}$ parameters

### EfficientNet: Compound Scaling

Scale depth, width, and resolution together:
$$\text{depth}: d = \alpha^\phi$$
$$\text{width}: w = \beta^\phi$$
$$\text{resolution}: r = \gamma^\phi$$

Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

---

## 7.7 Implementation

### Residual Block

```python
import numpy as np

class ResidualBlock:
    def __init__(self, in_channels, out_channels):
        self.conv1 = Conv2D(in_channels, out_channels, 3, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Shortcut if dimensions change
        if in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, 1)
        else:
            self.shortcut = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut:
            identity = self.shortcut(x)
        
        out = out + identity  # Skip connection
        out = relu(out)
        
        return out
```

### SE Block

```python
class SEBlock:
    def __init__(self, channels, reduction=16):
        self.fc1 = Linear(channels, channels // reduction)
        self.fc2 = Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch, channels, h, w = x.shape
        
        # Squeeze: Global average pooling
        y = x.mean(axis=(2, 3))  # (B, C)
        
        # Excitation
        y = relu(self.fc1(y))
        y = sigmoid(self.fc2(y))
        
        # Scale
        y = y.reshape(batch, channels, 1, 1)
        return x * y
```

---

## Exercises

### Exercise 7.1 (Section 7.2)
Implement a basic residual block and verify gradients flow through skip connections.

### Exercise 7.2 (Section 7.3)
Compare ResNet-18 with and without skip connections on CIFAR-10.

### Exercise 7.3 (Section 7.5)
Add SE blocks to a ResNet and measure accuracy improvement.

### Exercise 7.4 (Section 7.6)
Implement depthwise separable convolution and compare parameters with standard convolution.

---

## Summary

- Skip connections enable training very deep networks
- Residual learning: Learn $F(x) = H(x) - x$ instead of $H(x)$
- Dense connections maximize feature reuse
- SE blocks add channel attention
- Efficient architectures reduce parameters while maintaining accuracy

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 8.

### Key Papers
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition."
- Huang, G., et al. (2017). "Densely Connected Convolutional Networks."
- Hu, J., et al. (2018). "Squeeze-and-Excitation Networks."

---

## Course Progression

These architectural innovations appear in:
- **Week 9**: ResNet-style skip connections in CNNs
- **Week 11**: Residual connections in deep RNNs
- **Week 13**: Residual connections throughout Transformers
- **All modern architectures**: Skip connections are now standard

