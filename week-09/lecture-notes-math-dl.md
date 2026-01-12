# Week 9: Convolutional Neural Networks (CNNs)

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 9: Convolutional Neural Networks**.

---

## Overview

This module covers convolutional neural networks, the foundational architecture for computer vision and spatial data processing.

---

## Learning Objectives

- Understand the convolution operation mathematically
- Master CNN building blocks: convolution, pooling, padding
- Learn classic architectures: LeNet, AlexNet, VGG
- Connect CNN design to visual feature hierarchies

---

## 9.1 Motivation: Why Not Fully Connected?

### Problems with FC for Images

For a 224×224×3 image:
- Input size: 150,528 neurons
- First hidden layer (1000 neurons): 150 million parameters!
- No spatial structure preservation

### Solution: Convolutional Layers

1. **Parameter sharing**: Same filter applied everywhere
2. **Local connectivity**: Each neuron sees only a small region
3. **Translation equivariance**: Features detected regardless of position

---

## 9.2 The Convolution Operation

### 1D Convolution

For input $x$ and filter $w$ of size $k$:

$$(x * w)[i] = \sum_{j=0}^{k-1} x[i+j] \cdot w[j]$$

### 2D Convolution

For input $X$ and filter $W$ of size $k \times k$:

$$(X * W)[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m, j+n] \cdot W[m,n]$$

### Cross-Correlation (What CNNs Actually Use)

Most implementations use cross-correlation (no filter flip):

$$(X \star W)[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m, j+n] \cdot W[m,n]$$

The distinction doesn't matter since filters are learned.

### Multi-Channel Convolution

For input with $C_{in}$ channels and filter producing $C_{out}$ channels:

$$Y[:,:,c_{out}] = \sum_{c_{in}=1}^{C_{in}} X[:,:,c_{in}] * W[:,:,c_{in},c_{out}] + b[c_{out}]$$

**Filter dimensions**: $(k, k, C_{in}, C_{out})$

---

## 9.3 Convolution Parameters

### Stride

Step size when sliding the filter:

$$\text{Output size} = \left\lfloor \frac{n - k}{\text{stride}} \right\rfloor + 1$$

| Stride | Effect |
|--------|--------|
| 1 | Standard, preserves resolution |
| 2 | Downsamples by 2× |
| >2 | Aggressive downsampling |

### Padding

Add zeros around input to control output size:

$$\text{Output size} = \left\lfloor \frac{n + 2p - k}{\text{stride}} \right\rfloor + 1$$

| Padding Type | Formula | Effect |
|--------------|---------|--------|
| Valid | $p = 0$ | Output shrinks |
| Same | $p = \lfloor k/2 \rfloor$ | Output = Input (stride=1) |
| Full | $p = k - 1$ | Output grows |

### Dilation

Insert gaps in the filter to increase receptive field without more parameters:

$$\text{Effective kernel size} = k + (k-1) \times (d-1)$$

Where $d$ is dilation rate.

---

## 9.4 Pooling Layers

### Purpose

- Reduce spatial dimensions
- Provide translation invariance
- Reduce parameters and computation

### Max Pooling

$$Y[i,j] = \max_{m,n \in \text{window}} X[i \cdot s + m, j \cdot s + n]$$

### Average Pooling

$$Y[i,j] = \frac{1}{k^2} \sum_{m,n \in \text{window}} X[i \cdot s + m, j \cdot s + n]$$

### Global Average Pooling (GAP)

Average over entire spatial dimension:
$$Y[c] = \frac{1}{H \times W} \sum_{i,j} X[i,j,c]$$

Used before final classification layer.

### Comparison

| Type | Properties | Use |
|------|------------|-----|
| Max | Extracts strongest features | Most common |
| Average | Smoother, less aggressive | Some architectures |
| Global | No FC parameters | Modern CNNs |

---

## 9.5 Receptive Field

### Definition

The region of the input that affects a single output neuron.

### Calculation

For $L$ layers with kernel sizes $k_1, \ldots, k_L$ and strides $s_1, \ldots, s_L$:

$$R = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i=1}^{l-1} s_i$$

### Importance

- Larger receptive field → more context
- Deep networks → large receptive field
- Dilated convolutions → larger RF without more parameters

---

## 9.6 Classic Architectures

### LeNet-5 (1998)

First successful CNN for digit recognition.

```
Input(32×32×1) → Conv(5×5, 6) → Pool(2×2) → Conv(5×5, 16) → Pool(2×2) 
→ FC(120) → FC(84) → FC(10)
```

### AlexNet (2012)

ImageNet breakthrough.

Key innovations:
- ReLU activation
- Dropout
- Data augmentation
- GPU training

```
Conv(11×11, 96, s=4) → MaxPool → Conv(5×5, 256) → MaxPool 
→ Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → MaxPool 
→ FC(4096) → FC(4096) → FC(1000)
```

### VGGNet (2014)

Deeper with smaller filters.

Key insight: Two 3×3 convs = one 5×5 conv receptive field, fewer parameters.

```
VGG-16: 
[Conv3-64]×2 → Pool → [Conv3-128]×2 → Pool → [Conv3-256]×3 → Pool 
→ [Conv3-512]×3 → Pool → [Conv3-512]×3 → Pool → FC(4096)×2 → FC(1000)
```

### Modern Guidelines

| Era | Key Principle |
|-----|---------------|
| Early | Large filters (5×5, 7×7) |
| VGG | Small filters (3×3) stacked |
| Modern | 3×3 + skip connections |
| Efficient | Depthwise separable, compound scaling |

---

## 9.7 Convolution Backpropagation

### Forward

$$Y = X * W$$

### Backward

**Gradient w.r.t. weights**:
$$\frac{\partial L}{\partial W} = X * \frac{\partial L}{\partial Y}$$

**Gradient w.r.t. input** (transposed convolution):
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} *_{full} \text{flip}(W)$$

### Implementation Note

The backward pass for input is essentially a "deconvolution" or transposed convolution.

---

## 9.8 Implementation

### NumPy Convolution

```python
import numpy as np

def conv2d(X, W, stride=1, padding=0):
    """
    X: Input (H, W, C_in)
    W: Filters (k, k, C_in, C_out)
    """
    k = W.shape[0]
    H, W_in, C_in = X.shape
    C_out = W.shape[3]
    
    # Pad input
    if padding > 0:
        X = np.pad(X, ((padding, padding), (padding, padding), (0, 0)))
    
    H_out = (X.shape[0] - k) // stride + 1
    W_out = (X.shape[1] - k) // stride + 1
    
    output = np.zeros((H_out, W_out, C_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch
            patch = X[i*stride:i*stride+k, j*stride:j*stride+k, :]
            # Convolve with all filters
            for c in range(C_out):
                output[i, j, c] = np.sum(patch * W[:, :, :, c])
    
    return output

def max_pool2d(X, pool_size=2, stride=2):
    """
    X: Input (H, W, C)
    """
    H, W, C = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out, C))
    
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, :]
            output[i, j, :] = patch.max(axis=(0, 1))
    
    return output
```

### Simple CNN

```python
class SimpleCNN:
    def __init__(self):
        # Conv layer: 3x3, 1 input channel, 32 output channels
        self.conv1 = np.random.randn(3, 3, 1, 32) * 0.1
        self.conv2 = np.random.randn(3, 3, 32, 64) * 0.1
        self.fc = np.random.randn(64 * 7 * 7, 10) * 0.1
    
    def forward(self, X):
        # X: (28, 28, 1) for MNIST
        h1 = np.maximum(0, conv2d(X, self.conv1, padding=1))  # (28, 28, 32)
        h1 = max_pool2d(h1)  # (14, 14, 32)
        
        h2 = np.maximum(0, conv2d(h1, self.conv2, padding=1))  # (14, 14, 64)
        h2 = max_pool2d(h2)  # (7, 7, 64)
        
        h2_flat = h2.reshape(-1)  # (3136,)
        output = h2_flat @ self.fc  # (10,)
        
        return output
```

---

## 9.9 Visualizing CNN Features

### What CNNs Learn

| Layer | Features |
|-------|----------|
| Early (1-2) | Edges, colors, textures |
| Middle (3-4) | Parts, patterns |
| Deep (5+) | Objects, scenes |

### Visualization Methods

1. **Filter visualization**: Plot learned filters
2. **Activation maps**: Visualize layer outputs
3. **Gradient-based**: Saliency maps, Grad-CAM
4. **Feature inversion**: Generate images that maximize activations

---

## Exercises

### Exercise 9.1 (Section 9.2)
Manually compute the output of a 3×3 convolution on a 5×5 input with stride=1, padding=0.

### Exercise 9.2 (Section 9.3)
Calculate output dimensions for:
- Input: 224×224×3
- Conv: 7×7, 64 filters, stride=2, padding=3
- MaxPool: 3×3, stride=2

### Exercise 9.3 (Section 9.5)
Calculate the receptive field of VGG-16 at the final conv layer.

### Exercise 9.4 (Section 9.8)
Implement a simple CNN for MNIST classification.

---

## Summary

- Convolution provides parameter sharing and translation equivariance
- Stride controls downsampling, padding controls output size
- Pooling adds translation invariance
- Deeper networks learn hierarchical features
- Classic architectures: LeNet → AlexNet → VGG → ResNet

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 9.

### Key Papers
- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition."
- Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep CNNs."
- Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks."

---

## Course Progression

CNNs provide foundation for:
- **Week 10**: Advanced CNN architectures and transfer learning
- **Week 13**: Vision Transformers (ViT) compare to CNNs
- **Computer vision projects**: Image classification, object detection
- **Supplementary**: Compare with manual feature engineering (ML Refined Ch. 9)

