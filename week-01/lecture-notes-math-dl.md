# Week 1: Mathematical Foundations - Linear Algebra for Deep Learning

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapters 1-2: Linear Algebra Foundations**.

---

## Overview

This supplementary material introduces the essential linear algebra concepts required for understanding deep learning. While the primary text covers ML fundamentals, this module builds the mathematical toolkit needed for neural networks.

---

## Learning Objectives

- Understand vectors, matrices, and tensors as data representations
- Master matrix operations essential for neural networks
- Connect linear algebra to neural network computations
- Visualize transformations geometrically

---

## 1.1 Vectors and Their Operations

### Vectors as Data Representations

In deep learning, vectors represent:
- **Feature vectors**: Input data points
- **Weight vectors**: Learnable parameters
- **Activation vectors**: Layer outputs
- **Gradient vectors**: Directions of steepest descent

### Mathematical Definition

A vector $\mathbf{x} \in \mathbb{R}^n$ is an ordered collection of $n$ real numbers:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

### Vector Operations

| Operation | Definition | Deep Learning Context |
|-----------|------------|----------------------|
| Addition | $\mathbf{x} + \mathbf{y} = [x_i + y_i]$ | Combining features |
| Scalar mult. | $c\mathbf{x} = [cx_i]$ | Scaling activations |
| Dot product | $\mathbf{x}^T\mathbf{y} = \sum_i x_i y_i$ | Neuron computation |
| Norm | $\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$ | Regularization |

### The Dot Product: Foundation of Neural Networks

The dot product is the fundamental operation in neural networks:

$$\mathbf{w}^T \mathbf{x} = \sum_{i=1}^{n} w_i x_i$$

**Geometric interpretation**: 
- Measures alignment between vectors
- $\mathbf{w}^T \mathbf{x} = \|\mathbf{w}\| \|\mathbf{x}\| \cos\theta$

**Neural network interpretation**:
- Weighted sum of inputs
- Core computation of a single neuron

---

## 1.2 Matrices and Linear Transformations

### Matrices in Deep Learning

Matrices represent:
- **Weight matrices**: Connections between layers
- **Data batches**: Multiple samples processed together
- **Transformations**: Rotations, scalings, projections

### Matrix Definition

A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

### Matrix-Vector Multiplication

$$\mathbf{y} = \mathbf{A}\mathbf{x}$$

Each output component:
$$y_i = \sum_{j=1}^{n} a_{ij} x_j = \mathbf{a}_i^T \mathbf{x}$$

**Neural network interpretation**: One layer transformation
- $\mathbf{x}$: Input activations
- $\mathbf{A}$: Weight matrix
- $\mathbf{y}$: Output (before activation)

### Matrix-Matrix Multiplication

$$\mathbf{C} = \mathbf{A}\mathbf{B}$$

Where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times p}$, and $\mathbf{C} \in \mathbb{R}^{m \times p}$:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Deep learning context**: Batch processing - apply transformation to multiple inputs simultaneously.

---

## 1.3 Special Matrices

### Identity Matrix

$$\mathbf{I} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

Property: $\mathbf{I}\mathbf{x} = \mathbf{x}$ (skip connections in ResNets)

### Diagonal Matrix

$$\mathbf{D} = \text{diag}(d_1, d_2, \ldots, d_n)$$

Property: Scales each dimension independently (batch normalization scaling)

### Symmetric Matrix

$$\mathbf{A} = \mathbf{A}^T$$

Important for: Covariance matrices, Hessians in optimization

### Orthogonal Matrix

$$\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$$

Properties:
- Preserves lengths: $\|\mathbf{Qx}\| = \|\mathbf{x}\|$
- Preserves angles
- Used in orthogonal weight initialization

---

## 1.4 Eigenvalues and Eigenvectors

### Definition

For a square matrix $\mathbf{A}$, vector $\mathbf{v}$ is an eigenvector with eigenvalue $\lambda$ if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

### Geometric Interpretation

Eigenvectors define directions that are only scaled (not rotated) by the transformation.

### Eigendecomposition

For symmetric $\mathbf{A}$:
$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$

Where:
- $\mathbf{Q}$: Orthogonal matrix of eigenvectors
- $\mathbf{\Lambda}$: Diagonal matrix of eigenvalues

### Deep Learning Applications

| Application | Role of Eigenvalues/Eigenvectors |
|-------------|----------------------------------|
| PCA | Principal components are eigenvectors |
| Hessian analysis | Curvature directions |
| Weight initialization | Spectral norm control |
| Graph neural networks | Graph Laplacian eigenvectors |

---

## 1.5 Singular Value Decomposition (SVD)

### Definition

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: Left singular vectors (orthogonal)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: Diagonal matrix of singular values
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: Right singular vectors (orthogonal)

### Properties

- Singular values $\sigma_i \geq 0$, ordered $\sigma_1 \geq \sigma_2 \geq \cdots$
- $\text{rank}(\mathbf{A})$ = number of non-zero singular values

### Low-Rank Approximation

Truncated SVD with top $k$ singular values:
$$\mathbf{A}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

**Applications in deep learning**:
- Model compression
- Embedding dimension reduction
- Matrix factorization recommenders

---

## 1.6 Tensors: Multidimensional Arrays

### Definition

A tensor is a multidimensional array generalizing vectors (1D) and matrices (2D).

### Tensor Dimensions in Deep Learning

| Tensor Shape | Example Use |
|--------------|-------------|
| $(n,)$ | Feature vector |
| $(m, n)$ | Weight matrix, single image (grayscale) |
| $(h, w, c)$ | Single color image |
| $(b, h, w, c)$ | Batch of images |
| $(b, t, d)$ | Batch of sequences |

### Tensor Operations

```python
# Reshaping
x.reshape(batch, height, width, channels)

# Transposition
x.transpose(0, 2, 1)  # Swap dimensions

# Broadcasting
# Automatic expansion for element-wise ops
```

---

## 1.7 Norms and Distances

### Vector Norms

| Norm | Definition | Use in Deep Learning |
|------|------------|---------------------|
| L1 | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Sparsity (Lasso) |
| L2 | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Weight decay |
| L∞ | $\|\mathbf{x}\|_\infty = \max_i |x_i|$ | Adversarial robustness |

### Matrix Norms

| Norm | Definition | Use |
|------|------------|-----|
| Frobenius | $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ | Weight regularization |
| Spectral | $\|\mathbf{A}\|_2 = \sigma_{\max}$ | Lipschitz constraint |

---

## Code Implementation

```python
import numpy as np

# Vector operations
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

dot_product = np.dot(x, y)  # 32
l2_norm = np.linalg.norm(x)  # sqrt(14)

# Matrix operations
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 2, 3], [4, 5, 6]])
C = A @ B  # Matrix multiplication

# Eigendecomposition (symmetric)
S = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eigh(S)

# SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# Low-rank approximation
k = 1
A_approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
```

---

## Exercises

### Exercise 1.1 (Section 1.1)
For vectors $\mathbf{w} = [2, -1, 3]$ and $\mathbf{x} = [1, 4, -2]$:
1. Compute the dot product $\mathbf{w}^T\mathbf{x}$
2. Compute L2 norms of both vectors
3. Find the angle between them

### Exercise 1.2 (Section 1.2)
Given a weight matrix $\mathbf{W} \in \mathbb{R}^{128 \times 784}$ and a batch of 32 images flattened to vectors:
1. What is the shape of the input batch matrix?
2. What is the shape of the output?
3. How many multiply-add operations are needed?

### Exercise 1.3 (Section 1.5)
Implement SVD-based image compression:
1. Load a grayscale image as a matrix
2. Compute its SVD
3. Reconstruct using only top-k singular values for k = 5, 10, 50
4. Plot reconstruction error vs. k

---

## Summary

- Vectors and matrices are the language of neural networks
- Matrix multiplication implements layer transformations
- Eigendecomposition reveals transformation structure
- SVD enables dimensionality reduction and compression
- Tensors generalize to multi-dimensional data (images, sequences)

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapters 1-2.

### Supplementary Reading
- Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 2.
- Strang, G. (2019). *Linear Algebra and Learning from Data*.

---

## Course Progression

This foundational material prepares you for:
- **Week 2**: Calculus operations on vectors and matrices
- **Week 3**: Optimization algorithms using matrix computations
- **Week 4+**: Neural network layer computations as matrix operations
- **Week 8**: PCA as eigendecomposition (see supplementary ML Refined notes)

