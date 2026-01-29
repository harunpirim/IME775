# IME 775: Vectors, Matrices, and Tensors in Machine Learning

## Lecture Notes 3-4

**Course:** IME 775 - Mathematical Foundations of Deep Learning
**Reference:** Chaudhury, K. *Math and Architectures of Deep Learning*, Chapter 2

---

## 1. Introduction: Why Linear Algebra for Machine Learning?

At its core, machine learning—and indeed all computer software—is about number crunching. The fundamental challenge lies in organizing these numbers appropriately and grouping them into meaningful mathematical objects. This is where **vectors**, **matrices**, and **tensors** become indispensable.

### 1.1 The Central Role of Vectors

**Definition 2.1 (Vector):** A vector $\mathbf{x} \in \mathbb{R}^n$ is an ordered sequence of $n$ real numbers:
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

**Key Insight:** Every input and output in machine learning can be represented as a vector.

**Example (Object Recognition):** Consider a model that classifies images into {dog, human, cat}. The output is a probability vector:
$$\mathbf{y} = \begin{bmatrix} P(\text{dog}) \\ P(\text{human}) \\ P(\text{cat}) \end{bmatrix} = \begin{bmatrix} y_0 \\ y_1 \\ y_2 \end{bmatrix}$$

where $y_i \geq 0$ and $\sum_i y_i = 1$.

### 1.2 The Geometric View of Vectors

**Theorem 2.1 (Geometric Interpretation):** A vector $\mathbf{x} \in \mathbb{R}^n$ represents a point in $n$-dimensional Euclidean space.

**Implication for ML:** A machine learning model can be viewed as a **geometric transformation** that maps input points to output points in high-dimensional space.

**Critical Requirement:** All vectors used in a machine learning computation must consistently use the same coordinate system or be transformed appropriately.

---

## 2. Matrices and Their Role in Machine Learning

### 2.1 Training Data as Matrices

**Definition 2.2 (Data Matrix):** Given $m$ training instances, each with $n$ features, we represent the training data as a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$:
$$\mathbf{X} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}$$

Each row $\mathbf{x}_i^T$ corresponds to a single training instance.

### 2.2 Digital Images as Matrices

**Definition 2.3 (Grayscale Image):** A grayscale image with height $H$ and width $W$ is represented as $\mathbf{I} \in \mathbb{Z}^{H \times W}$ where:
- $I_{ij} \in [0, 255]$
- $I_{ij} = 0$ represents black
- $I_{ij} = 255$ represents white
- $I_{ij} = 128$ represents mid-gray

**Example:** A 4×9 pixel image:
$$\mathbf{I} = \begin{bmatrix}
0 & 28 & 56 & 85 & 113 & 141 & 170 & 198 & 226 \\
9 & 37 & 66 & 94 & 122 & 151 & 179 & 207 & 235 \\
18 & 47 & 75 & 103 & 132 & 160 & 188 & 216 & 245 \\
28 & 56 & 85 & 113 & 141 & 170 & 198 & 226 & 255
\end{bmatrix}$$

---

## 3. Tensors: Multidimensional Arrays

**Definition 2.4 (Tensor):** A tensor is a multidimensional array that generalizes scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions.

**Example (Image Batch):** A batch of 64 RGB images of size 224×224 is a 4D tensor:
$$\mathcal{T} \in \mathbb{R}^{64 \times 3 \times 224 \times 224}$$

**Dimension ordering (PyTorch convention):** `[batch_size, channels, height, width]`

---

## 4. Fundamental Vector Operations

### 4.1 The Transpose Operation

**Definition 2.5 (Vector Transpose):** For a column vector $\mathbf{x} \in \mathbb{R}^n$:
$$\mathbf{x}^T = \begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix}$$

**Property:** $(\mathbf{x}^T)^T = \mathbf{x}$

### 4.2 The Dot Product (Inner Product)

**Definition 2.6 (Dot Product):** For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T\mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1v_1 + u_2v_2 + \cdots + u_nv_n$$

**Note:** The dot product is defined only for vectors of the same dimension.

**PyTorch Implementation:**
```python
import torch
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(u, v)  # Returns 32.0
```

### 4.3 L2 Norm (Euclidean Length)

**Definition 2.7 (L2 Norm):** The L2 norm of $\mathbf{x} \in \mathbb{R}^n$ is:
$$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T\mathbf{x}} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**Properties:**
1. $\|\mathbf{x}\|_2 \geq 0$ with equality iff $\mathbf{x} = \mathbf{0}$
2. $\|c\mathbf{x}\|_2 = |c| \|\mathbf{x}\|_2$ for scalar $c$
3. $\|\mathbf{x} + \mathbf{y}\|_2 \leq \|\mathbf{x}\|_2 + \|\mathbf{y}\|_2$ (Triangle Inequality)

**ML Application:** The L2 norm is used to measure model error:
$$\text{MSE} = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$$

### 4.4 Geometric Interpretation of the Dot Product

**Theorem 2.2:** For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta$$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$.

**Corollaries:**
1. $\mathbf{u} \cdot \mathbf{v} > 0 \Rightarrow \theta < 90°$ (vectors point in similar directions)
2. $\mathbf{u} \cdot \mathbf{v} < 0 \Rightarrow \theta > 90°$ (vectors point in opposite directions)
3. $\mathbf{u} \cdot \mathbf{v} = 0 \Rightarrow \theta = 90°$ (vectors are orthogonal)

### 4.5 Cosine Similarity

**Definition 2.8 (Cosine Similarity):** For non-zero vectors $\mathbf{u}, \mathbf{v}$:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

**Properties:**
- $\cos(\mathbf{u}, \mathbf{v}) \in [-1, 1]$
- $\cos(\mathbf{u}, \mathbf{v}) = 1$: vectors are identical in direction
- $\cos(\mathbf{u}, \mathbf{v}) = 0$: vectors are orthogonal
- $\cos(\mathbf{u}, \mathbf{v}) = -1$: vectors point in opposite directions

**ML Application:** Document similarity, word embeddings, recommendation systems.

---

## 5. Matrix Operations

### 5.1 Matrix Transpose

**Definition 2.9 (Matrix Transpose):** For $\mathbf{A} \in \mathbb{R}^{m \times n}$, the transpose $\mathbf{A}^T \in \mathbb{R}^{n \times m}$ satisfies:
$$(\mathbf{A}^T)_{ij} = A_{ji}$$

**Properties:**
1. $(\mathbf{A}^T)^T = \mathbf{A}$
2. $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
3. $(c\mathbf{A})^T = c\mathbf{A}^T$
4. $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$ (Reversal Law)

### 5.2 Matrix-Vector Multiplication

**Definition 2.10:** For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$:
$$\mathbf{y} = \mathbf{Ax} \in \mathbb{R}^m \quad \text{where} \quad y_i = \sum_{j=1}^{n} A_{ij}x_j$$

**Interpretation:** Each element $y_i$ is the dot product of the $i$-th row of $\mathbf{A}$ with $\mathbf{x}$.

### 5.3 Matrix-Matrix Multiplication

**Definition 2.11:** For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$:
$$\mathbf{C} = \mathbf{AB} \in \mathbb{R}^{m \times p} \quad \text{where} \quad C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

**Compatibility Rule:** Matrix multiplication $\mathbf{AB}$ is defined only when:
$$\text{cols}(\mathbf{A}) = \text{rows}(\mathbf{B})$$

**Non-Commutativity:** In general, $\mathbf{AB} \neq \mathbf{BA}$.

---

## 6. Hyperlines and Hyperplanes

### 6.1 Parametric Equation of a Line

**Theorem 2.3 (Parametric Line):** Any point on the line passing through $\mathbf{p}$ and $\mathbf{q}$ can be written as:
$$\mathbf{r}(\alpha) = \alpha\mathbf{p} + (1-\alpha)\mathbf{q}$$

**Interpretation:**
- $\alpha \in [0, 1]$: points between $\mathbf{p}$ and $\mathbf{q}$
- $\alpha < 0$: points beyond $\mathbf{q}$ (away from $\mathbf{p}$)
- $\alpha > 1$: points beyond $\mathbf{p}$ (away from $\mathbf{q}$)

### 6.2 Hyperplanes and Classification

**Definition 2.12 (Hyperplane):** In $\mathbb{R}^n$, a hyperplane is defined by:
$$\mathbf{w}^T\mathbf{x} + b = 0$$

where $\mathbf{w} \in \mathbb{R}^n$ is the normal vector and $b$ is the bias.

**ML Application:** Linear classifiers separate classes using hyperplanes:
$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$$

---

## 7. Linear Combinations and Vector Spaces

### 7.1 Linear Combinations

**Definition 2.13 (Linear Combination):** A linear combination of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is:
$$\mathbf{u} = \sum_{i=1}^{k} \alpha_i \mathbf{v}_i = \alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_k\mathbf{v}_k$$

where $\alpha_i \in \mathbb{R}$ are scalar coefficients.

### 7.2 Span and Basis

**Definition 2.14 (Span):** The span of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is the set of all possible linear combinations:
$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{\sum_{i=1}^{k} \alpha_i \mathbf{v}_i : \alpha_i \in \mathbb{R}\right\}$$

**Definition 2.15 (Basis):** A set of vectors $\{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ is a basis for $\mathbb{R}^n$ if:
1. They are linearly independent
2. They span $\mathbb{R}^n$

### 7.3 Linear Independence

**Definition 2.16:** Vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are linearly independent iff:
$$\sum_{i=1}^{k} \alpha_i \mathbf{v}_i = \mathbf{0} \implies \alpha_i = 0 \text{ for all } i$$

---

## 8. Linear Transforms

### 8.1 Definition and Properties

**Definition 2.17 (Linear Transform):** A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is linear iff for all $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ and $c \in \mathbb{R}$:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (Additivity)
2. $T(c\mathbf{u}) = cT(\mathbf{u})$ (Homogeneity)

**Theorem 2.4 (Matrix Representation):** Every linear transform $T: \mathbb{R}^n \to \mathbb{R}^m$ can be represented by matrix multiplication:
$$T(\mathbf{x}) = \mathbf{Ax}$$

for some $\mathbf{A} \in \mathbb{R}^{m \times n}$.

**Corollary:** In finite dimensions, matrix multiplication and linear transformation are equivalent.

### 8.2 Collinearity Preservation

**Theorem 2.5:** Linear transforms preserve collinearity—points on a line map to points on a line.

**Proof Sketch:** If $\mathbf{r}(\alpha) = \alpha\mathbf{p} + (1-\alpha)\mathbf{q}$ lies on a line, then:
$$T(\mathbf{r}(\alpha)) = \alpha T(\mathbf{p}) + (1-\alpha)T(\mathbf{q})$$
which also describes a line through $T(\mathbf{p})$ and $T(\mathbf{q})$. $\square$

---

## 9. Linear Systems and Matrix Inverse

### 9.1 Linear Systems

**Definition 2.18 (Linear System):** A system of $m$ linear equations in $n$ unknowns:
$$\mathbf{Ax} = \mathbf{b}$$

where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

### 9.2 Matrix Inverse

**Definition 2.19 (Matrix Inverse):** For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the inverse $\mathbf{A}^{-1}$ satisfies:
$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Properties:**
1. $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
2. $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
3. $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$

### 9.3 Determinants and Singularity

**Definition 2.20 (Determinant):** For $\mathbf{A} \in \mathbb{R}^{n \times n}$, the determinant $\det(\mathbf{A})$ is a scalar that characterizes the matrix.

**Theorem 2.6 (Invertibility Criterion):** $\mathbf{A}$ is invertible $\iff \det(\mathbf{A}) \neq 0$

**Equivalent Conditions for Singularity:**
- $\det(\mathbf{A}) = 0$
- $\mathbf{A}$ has linearly dependent rows/columns
- $\mathbf{A}^{-1}$ does not exist
- $\mathbf{Ax} = \mathbf{b}$ may have no solution or infinitely many solutions

### 9.4 Over- and Under-Determined Systems

**Case 1: Overdetermined ($m > n$):** More equations than unknowns. Typically no exact solution exists due to noise. We seek the least-squares solution:
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{Ax} - \mathbf{b}\|_2^2$$

**Case 2: Underdetermined ($m < n$):** Fewer equations than unknowns. Infinitely many solutions exist. We typically seek the minimum-norm solution:
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{x}\|_2 \quad \text{subject to} \quad \mathbf{Ax} = \mathbf{b}$$

---

## 10. Moore-Penrose Pseudo-Inverse

### 10.1 Definition

**Definition 2.21 (Pseudo-Inverse):** For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the Moore-Penrose pseudo-inverse $\mathbf{A}^+ \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying:

1. $\mathbf{A}\mathbf{A}^+\mathbf{A} = \mathbf{A}$
2. $\mathbf{A}^+\mathbf{A}\mathbf{A}^+ = \mathbf{A}^+$
3. $(\mathbf{A}\mathbf{A}^+)^T = \mathbf{A}\mathbf{A}^+$
4. $(\mathbf{A}^+\mathbf{A})^T = \mathbf{A}^+\mathbf{A}$

### 10.2 Computation Formulas

**For overdetermined systems ($m > n$, full column rank):**
$$\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

**For underdetermined systems ($m < n$, full row rank):**
$$\mathbf{A}^+ = \mathbf{A}^T(\mathbf{A}\mathbf{A}^T)^{-1}$$

### 10.3 Geometric Intuition

The pseudo-inverse provides:
- **Overdetermined:** The closest point (in L2 sense) to $\mathbf{b}$ in the column space of $\mathbf{A}$
- **Underdetermined:** The smallest norm solution among all solutions

**PyTorch Implementation:**
```python
import torch
A = torch.randn(5, 3)  # Overdetermined system
b = torch.randn(5)
x = torch.linalg.lstsq(A, b).solution  # Least-squares solution
```

---

## 11. Eigenvalues and Eigenvectors

### 11.1 Fundamental Definition

**Definition 2.22 (Eigenpair):** For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ and non-zero vector $\mathbf{v}$ form an eigenpair if:
$$\mathbf{Av} = \lambda\mathbf{v}$$

- $\lambda$ is an **eigenvalue**
- $\mathbf{v}$ is the corresponding **eigenvector**

### 11.2 Geometric Interpretation

**Intuition:** Eigenvectors are directions that remain unchanged (up to scaling) under the linear transform $\mathbf{A}$. The eigenvalue $\lambda$ gives the scaling factor.

**Example (Rotation about Z-axis):** For a 3D rotation matrix about the Z-axis:
- The Z-axis is an eigenvector with eigenvalue $\lambda = 1$
- Points on the axis of rotation do not move

### 11.3 Properties of Eigenvectors

**Theorem 2.7:** Eigenvectors corresponding to distinct eigenvalues are linearly independent.

**Proof:** Suppose $\mathbf{Av}_1 = \lambda_1\mathbf{v}_1$ and $\mathbf{Av}_2 = \lambda_2\mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$.
Assume $\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 = \mathbf{0}$.
Applying $\mathbf{A}$: $\alpha_1\lambda_1\mathbf{v}_1 + \alpha_2\lambda_2\mathbf{v}_2 = \mathbf{0}$.
Subtracting: $\alpha_2(\lambda_2 - \lambda_1)\mathbf{v}_2 = \mathbf{0}$.
Since $\lambda_1 \neq \lambda_2$ and $\mathbf{v}_2 \neq \mathbf{0}$, we have $\alpha_2 = 0$. Similarly, $\alpha_1 = 0$. $\square$

### 11.4 Computing Eigenvalues

**Characteristic Equation:** Eigenvalues satisfy:
$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

This yields a polynomial of degree $n$ in $\lambda$.

**PyTorch Implementation:**
```python
A = torch.randn(3, 3)
eigenvalues, eigenvectors = torch.linalg.eig(A)
# For symmetric matrices (real eigenvalues):
eigenvalues, eigenvectors = torch.linalg.eigh(A)
```

---

## 12. Orthogonal Matrices

### 12.1 Definition

**Definition 2.23 (Orthogonal Matrix):** $\mathbf{Q} \in \mathbb{R}^{n \times n}$ is orthogonal iff:
$$\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$$

Equivalently: $\mathbf{Q}^{-1} = \mathbf{Q}^T$

### 12.2 Properties

1. **Norm Preservation:** $\|\mathbf{Qx}\|_2 = \|\mathbf{x}\|_2$
2. **Angle Preservation:** $\cos(\mathbf{Qx}, \mathbf{Qy}) = \cos(\mathbf{x}, \mathbf{y})$
3. **Determinant:** $\det(\mathbf{Q}) = \pm 1$
   - $\det(\mathbf{Q}) = +1$: rotation
   - $\det(\mathbf{Q}) = -1$: reflection

### 12.3 Rotation Matrices

**2D Rotation by angle $\theta$:**
$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Eigenvalues of Rotation:** $\lambda = e^{\pm i\theta} = \cos\theta \pm i\sin\theta$

The eigenvalues encode the rotation angle.

---

## 13. Matrix Diagonalization

### 13.1 Diagonalizable Matrices

**Definition 2.24:** A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is diagonalizable if there exists an invertible matrix $\mathbf{S}$ such that:
$$\mathbf{A} = \mathbf{S}\boldsymbol{\Lambda}\mathbf{S}^{-1}$$

where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains the eigenvalues.

**Construction:** The columns of $\mathbf{S}$ are the eigenvectors of $\mathbf{A}$.

### 13.2 Applications of Diagonalization

**Computing Matrix Powers:**
$$\mathbf{A}^k = \mathbf{S}\boldsymbol{\Lambda}^k\mathbf{S}^{-1} = \mathbf{S}\text{diag}(\lambda_1^k, \ldots, \lambda_n^k)\mathbf{S}^{-1}$$

**Solving Linear Systems:**
$$\mathbf{Ax} = \mathbf{b} \implies \mathbf{x} = \mathbf{S}\boldsymbol{\Lambda}^{-1}\mathbf{S}^{-1}\mathbf{b}$$

### 13.3 Spectral Theorem

**Theorem 2.8 (Spectral Theorem):** For a symmetric matrix $\mathbf{A} = \mathbf{A}^T$:
1. All eigenvalues are real
2. Eigenvectors are orthogonal
3. $\mathbf{A}$ has the spectral decomposition:
$$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i\mathbf{q}_i^T$$

where $\mathbf{Q}$ is orthogonal and columns $\mathbf{q}_i$ are orthonormal eigenvectors.

**PyTorch Implementation:**
```python
A = torch.randn(3, 3)
A = (A + A.T) / 2  # Make symmetric
eigenvalues, Q = torch.linalg.eigh(A)
# Reconstruct: A_reconstructed = Q @ torch.diag(eigenvalues) @ Q.T
```

---

## 14. Applications to Machine Learning

### 14.1 Principal Component Analysis (PCA)

**Goal:** Reduce dimensionality by projecting data onto directions of maximum variance.

**Algorithm:**
1. Center the data: $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{x}}$
2. Compute covariance matrix: $\mathbf{C} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$
3. Compute eigenvectors of $\mathbf{C}$
4. Project onto top $k$ eigenvectors (principal components)

### 14.2 Singular Value Decomposition (SVD)

**Theorem 2.9:** Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:
$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal (left singular vectors)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal (singular values)
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: orthogonal (right singular vectors)

**Connection to Pseudo-Inverse:**
$$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$$

---

## 15. Summary: Key Concepts for ML

| Concept | Definition | ML Application |
|---------|------------|----------------|
| Vector | Ordered sequence of numbers | Features, embeddings |
| Matrix | 2D array of numbers | Training data, transformations |
| Tensor | Multidimensional array | Image batches, neural network parameters |
| Dot Product | $\sum_i u_i v_i$ | Similarity measures |
| Cosine Similarity | Normalized dot product | Document retrieval |
| L2 Norm | $\sqrt{\sum_i x_i^2}$ | Loss functions |
| Linear Transform | $T(\mathbf{x}) = \mathbf{Ax}$ | Neural network layers |
| Pseudo-Inverse | Generalized inverse | Least-squares solutions |
| Eigendecomposition | $\mathbf{A} = \mathbf{S}\boldsymbol{\Lambda}\mathbf{S}^{-1}$ | PCA, spectral methods |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | Dimensionality reduction |

---

## References

1. Chaudhury, K. (2024). *Math and Architectures of Deep Learning*. Manning Publications.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press.

---

*IME 775 Lecture Notes | Mathematical Foundations of Deep Learning*
