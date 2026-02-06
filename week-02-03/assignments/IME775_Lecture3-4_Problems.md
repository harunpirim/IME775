# IME 775: Problem Set - Vectors, Matrices, and Tensors

## Lecture Notes 3-4 Practice Problems

**Course:** IME 775 - Mathematical Foundations of Deep Learning
**Reference:** Chaudhury, K. *Math and Architectures of Deep Learning*, Chapter 2

---

## Part I: Vectors and Basic Operations

### Problem 1: Feature Vector Representation
A document retrieval system represents documents using TF-IDF vectors. Consider three documents with the following term frequency vectors for the vocabulary {machine, learning, deep, neural, network}:

$$\mathbf{d}_1 = \begin{bmatrix} 5 \\ 3 \\ 2 \\ 1 \\ 4 \end{bmatrix}, \quad
\mathbf{d}_2 = \begin{bmatrix} 4 \\ 4 \\ 3 \\ 2 \\ 3 \end{bmatrix}, \quad
\mathbf{d}_3 = \begin{bmatrix} 1 \\ 1 \\ 5 \\ 4 \\ 2 \end{bmatrix}$$

**(a)** Compute $\|\mathbf{d}_1\|_2$, $\|\mathbf{d}_2\|_2$, and $\|\mathbf{d}_3\|_2$.

**(b)** Calculate the cosine similarity between all pairs: $\cos(\mathbf{d}_1, \mathbf{d}_2)$, $\cos(\mathbf{d}_1, \mathbf{d}_3)$, $\cos(\mathbf{d}_2, \mathbf{d}_3)$.

**(c)** Based on cosine similarity, which two documents are most similar? Explain why cosine similarity is preferred over Euclidean distance for document comparison.

**(d)** A query vector is $\mathbf{q} = [3, 2, 4, 3, 1]^T$. Rank the documents by relevance to this query using cosine similarity.

---

### Problem 2: Orthogonality and Linear Independence
Consider the vectors:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \quad
\mathbf{v}_2 = \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix}, \quad
\mathbf{v}_3 = \begin{bmatrix} -1 \\ 1 \\ 3 \end{bmatrix}$$

**(a)** Verify whether $\mathbf{v}_1 \perp \mathbf{v}_2$ (i.e., check if they are orthogonal).

**(b)** Are $\mathbf{v}_1$, $\mathbf{v}_2$, $\mathbf{v}_3$ linearly independent? Justify your answer by attempting to find scalars $\alpha_1, \alpha_2, \alpha_3$ such that $\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \alpha_3\mathbf{v}_3 = \mathbf{0}$.

**(c)** Apply the Gram-Schmidt process to $\mathbf{v}_1$ and $\mathbf{v}_2$ to create an orthonormal pair $\{\mathbf{u}_1, \mathbf{u}_2\}$.

---

### Problem 3: Dot Product Geometry
Let $\mathbf{a} = [3, 4]^T$ and $\mathbf{b} = [b_1, b_2]^T$.

**(a)** Find all vectors $\mathbf{b}$ with $\|\mathbf{b}\|_2 = 5$ that are orthogonal to $\mathbf{a}$.

**(b)** Find the vector $\mathbf{b}$ with $\|\mathbf{b}\|_2 = 5$ that maximizes $\mathbf{a} \cdot \mathbf{b}$.

**(c)** What is the geometric interpretation of $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2}$ when $\mathbf{b}$ is a unit vector?

---

## Part II: Matrix Operations and Transforms

### Problem 4: Linear Transformations
Consider the transformation matrix:
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$$

**(a)** Apply $\mathbf{A}$ to the standard basis vectors $\mathbf{e}_1 = [1, 0]^T$ and $\mathbf{e}_2 = [0, 1]^T$. What do the columns of $\mathbf{A}$ represent geometrically?

**(b)** Transform the unit square with vertices at $(0,0)$, $(1,0)$, $(1,1)$, $(0,1)$. Sketch the resulting shape.

**(c)** Compute $\det(\mathbf{A})$ and interpret its meaning in terms of area scaling.

**(d)** Describe the geometric effect of this transformation (e.g., stretch, shear, rotation).

---

### Problem 5: Matrix Multiplication and Composition
Let:
$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \quad
\mathbf{S} = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$$

where $\mathbf{R}$ is a rotation matrix and $\mathbf{S}$ is a scaling matrix.

**(a)** For $\theta = 45°$, compute $\mathbf{RS}$ (scale then rotate) and $\mathbf{SR}$ (rotate then scale).

**(b)** Show that $\mathbf{RS} \neq \mathbf{SR}$ in general. When would these be equal?

**(c)** Prove that $\mathbf{R}^T\mathbf{R} = \mathbf{I}$ (i.e., $\mathbf{R}$ is orthogonal). What does this imply about $\mathbf{R}^{-1}$?

**(d)** A neural network layer applies $\mathbf{y} = \mathbf{W}_2(\mathbf{W}_1\mathbf{x})$. Why can't this be simplified to $\mathbf{y} = \mathbf{W}\mathbf{x}$ for some single matrix $\mathbf{W}$ when nonlinear activations are involved?

---

### Problem 6: Image Transformations
A grayscale image is represented as a $3 \times 3$ matrix:
$$\mathbf{I} = \begin{bmatrix} 100 & 150 & 200 \\ 120 & 170 & 220 \\ 140 & 190 & 240 \end{bmatrix}$$

**(a)** What is the transposed image $\mathbf{I}^T$? Describe the geometric effect.

**(b)** Represent this image as a flattened vector (row-major order). What is the dimension of this vector?

**(c)** How would you represent a batch of 64 RGB images of size $224 \times 224$ as a tensor? Specify the tensor dimensions using PyTorch convention.

---

## Part III: Linear Systems and Matrix Inverse

### Problem 7: Solving Linear Systems
Consider the system $\mathbf{Ax} = \mathbf{b}$ where:
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 5 \\ 11 \end{bmatrix}$$

**(a)** Compute $\det(\mathbf{A})$. Is $\mathbf{A}$ invertible?

**(b)** Find $\mathbf{A}^{-1}$ using the formula for $2 \times 2$ matrices:
$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})}\begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix}$$

**(c)** Solve for $\mathbf{x}$ using $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$.

**(d)** Verify your solution by computing $\mathbf{Ax}$.

---

### Problem 8: Singular Matrices and Linear Dependence
Let:
$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 2 \end{bmatrix}$$

**(a)** Show that $\mathbf{A}$ is singular by finding linearly dependent rows.

**(b)** For what vector $\mathbf{b}$ does $\mathbf{Ax} = \mathbf{b}$ have a solution? (Hint: $\mathbf{b}$ must be in the column space of $\mathbf{A}$)

**(c)** In machine learning, what problem does a singular (or near-singular) design matrix indicate?

---

### Problem 9: Overdetermined Systems and Least Squares
A linear regression model $y = w_0 + w_1x$ is fitted to 4 data points:

| $x$ | $y$ |
|-----|-----|
| 0   | 1   |
| 1   | 2.5 |
| 2   | 4.2 |
| 3   | 5.8 |

**(a)** Write this as an overdetermined system $\mathbf{X}\boldsymbol{w} = \mathbf{y}$ where $\boldsymbol{w} = [w_0, w_1]^T$. What are the dimensions of $\mathbf{X}$?

**(b)** Compute $\mathbf{X}^T\mathbf{X}$ and $\mathbf{X}^T\mathbf{y}$.

**(c)** Solve the normal equations $(\mathbf{X}^T\mathbf{X})\boldsymbol{w} = \mathbf{X}^T\mathbf{y}$ to find the least-squares solution.

**(d)** Compute the predicted values $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{w}$ and the residuals $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$.

**(e)** What is the Mean Squared Error (MSE)?

---

## Part IV: Eigenvalues and Eigenvectors

### Problem 10: Computing Eigenvalues
Find the eigenvalues and eigenvectors of:
$$\mathbf{A} = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$$

**(a)** Write and solve the characteristic equation $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$.

**(b)** For each eigenvalue, find the corresponding eigenvector by solving $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$.

**(c)** Verify that $\mathbf{Av} = \lambda\mathbf{v}$ for each eigenpair.

**(d)** Are the eigenvectors orthogonal? Why or why not?

---

### Problem 11: Symmetric Matrices and the Spectral Theorem
Consider the symmetric matrix:
$$\mathbf{A} = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

**(a)** Find the eigenvalues and eigenvectors of $\mathbf{A}$.

**(b)** Verify that the eigenvectors are orthogonal.

**(c)** Construct the orthogonal matrix $\mathbf{Q}$ (columns are normalized eigenvectors) and diagonal matrix $\boldsymbol{\Lambda}$.

**(d)** Verify that $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$.

**(e)** Express $\mathbf{A}$ as a sum of outer products: $\mathbf{A} = \sum_i \lambda_i \mathbf{q}_i\mathbf{q}_i^T$.

---

### Problem 12: Matrix Powers via Diagonalization
Let:
$$\mathbf{A} = \begin{bmatrix} 0.9 & 0.2 \\ 0.1 & 0.8 \end{bmatrix}$$

This represents a Markov chain transition matrix.

**(a)** Find the eigenvalues and eigenvectors of $\mathbf{A}$.

**(b)** Write $\mathbf{A}$ in diagonalized form: $\mathbf{A} = \mathbf{S}\boldsymbol{\Lambda}\mathbf{S}^{-1}$.

**(c)** Compute $\mathbf{A}^{10}$ using $\mathbf{A}^{10} = \mathbf{S}\boldsymbol{\Lambda}^{10}\mathbf{S}^{-1}$.

**(d)** What happens as $k \to \infty$? What is the steady-state distribution?

**(e)** In the context of RNNs, what does the spectral radius $\rho(\mathbf{A}) = \max_i |\lambda_i|$ tell us about gradient flow?

---

### Problem 13: Principal Component Analysis
Given centered data points:
$$\mathbf{X} = \begin{bmatrix} -2 & -1 \\ -1 & 0 \\ 0 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix}$$

**(a)** Compute the covariance matrix $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$.

**(b)** Find the eigenvalues and eigenvectors of $\mathbf{C}$.

**(c)** What percentage of variance does PC1 capture?

**(d)** Project the data onto PC1. What are the new 1D coordinates?

**(e)** Reconstruct the data from the 1D projection. What is the reconstruction error?

---

## Part V: Hyperplanes and Classification

### Problem 14: Decision Boundaries
A linear classifier uses the decision rule:
$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$$

where $\mathbf{w} = [2, 1]^T$ and $b = -3$.

**(a)** Write the equation of the decision boundary (hyperplane).

**(b)** Classify the points: $\mathbf{x}_1 = [1, 2]^T$, $\mathbf{x}_2 = [2, 0]^T$, $\mathbf{x}_3 = [0, 1]^T$.

**(c)** Compute the signed distance from each point to the hyperplane. (Formula: $d = \frac{\mathbf{w}^T\mathbf{x} + b}{\|\mathbf{w}\|_2}$)

**(d)** What is the geometric interpretation of the weight vector $\mathbf{w}$?

---

### Problem 15: Parametric Lines and Interpolation
Consider two points $\mathbf{p} = [1, 2, 3]^T$ and $\mathbf{q} = [4, 5, 6]^T$.

**(a)** Write the parametric equation of the line passing through $\mathbf{p}$ and $\mathbf{q}$.

**(b)** Find the point on the line when $\alpha = 0.5$ (midpoint).

**(c)** For what value of $\alpha$ does the line intersect the plane $x_1 + x_2 + x_3 = 12$?

**(d)** In machine learning, how is this parametric form used in linear interpolation between embeddings?

---

## Part VI: PyTorch Implementation

### Problem 16: Implementing Matrix Operations
Write PyTorch code to:

**(a)** Create two random tensors of shape $(3, 4)$ and $(4, 2)$, and compute their matrix product.

**(b)** Compute the eigenvalues and eigenvectors of a random symmetric $4 \times 4$ matrix.

**(c)** Solve the least-squares problem $\mathbf{Ax} = \mathbf{b}$ where $\mathbf{A}$ is $10 \times 3$ and $\mathbf{b}$ is $10 \times 1$.

**(d)** Implement a function that computes cosine similarity between two vectors.

---

### Problem 17: Numerical Stability
**(a)** Create a nearly singular matrix and attempt to compute its inverse. What happens?

**(b)** Compare the results of solving $\mathbf{Ax} = \mathbf{b}$ using:
   - Direct inversion: $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$
   - `torch.linalg.solve(A, b)`
   - `torch.linalg.lstsq(A, b)`

**(c)** When should you use the pseudo-inverse instead of the regular inverse?

---

## Part VII: Conceptual Questions

### Problem 18: True or False (Justify)
Determine whether each statement is true or false and provide justification:

**(a)** If $\mathbf{AB} = \mathbf{BA}$, then both $\mathbf{A}$ and $\mathbf{B}$ must be diagonal matrices.

**(b)** The eigenvalues of a symmetric matrix are always real.

**(c)** If $\det(\mathbf{A}) = 1$, then $\mathbf{A}$ preserves area.

**(d)** Two orthogonal vectors always have cosine similarity of exactly 0.

**(e)** The pseudo-inverse $\mathbf{A}^+$ always exists, even when $\mathbf{A}^{-1}$ does not.

**(f)** Linear transforms preserve collinearity (points on a line map to points on a line).

---

### Problem 19: Connections to Deep Learning
**(a)** How does a fully-connected neural network layer $\mathbf{y} = \sigma(\mathbf{Wx} + \mathbf{b})$ relate to linear transformations?

**(b)** Why is eigenvalue analysis important for understanding gradient flow in recurrent neural networks?

**(c)** How does PCA relate to dimensionality reduction in autoencoders?

**(d)** Explain why batch normalization involves computing covariance statistics.

**(e)** In what sense is the attention mechanism in transformers related to computing similarities via dot products?

---

### Problem 20: Proof Exercise
Prove the following:

**(a)** For any matrix $\mathbf{A}$, prove that $\mathbf{A}^T\mathbf{A}$ is symmetric.

**(b)** Prove that if $\mathbf{v}$ is an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda$, then $\mathbf{v}$ is also an eigenvector of $\mathbf{A}^2$ with eigenvalue $\lambda^2$.

**(c)** Prove that eigenvectors corresponding to distinct eigenvalues of a symmetric matrix are orthogonal.

---

## Answer Key Hints

For computational problems, verify your answers using PyTorch:

```python
import torch

# Example verification for Problem 7
A = torch.tensor([[2., 1.], [4., 3.]])
b = torch.tensor([5., 11.])
x = torch.linalg.solve(A, b)
print(f"Solution: {x}")
print(f"Verification A@x: {A @ x}")
```

---

*IME 775 - Mathematical Foundations of Deep Learning*
*Practice Problems for Lectures 3-4*
