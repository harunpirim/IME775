# IME 775 — Lecture 6
## Quadratic Forms, Positive Definiteness, and Matrix Norms

---

## Chapter 4: Finding Structure in Data

**Core Challenge:** High-dimensional data often lies near lower-dimensional subspaces.

**Tools we'll learn:**
- Quadratic forms → understand loss surfaces
- Positive definiteness → convex optimization
- Matrix norms → measure approximation quality
- PCA & SVD → dimensionality reduction

---

## Quadratic Forms: Definition

**Definition:** For symmetric matrix $A$, the quadratic form is:

$$Q(\vec{x}) = \vec{x}^T A \vec{x}$$

&nbsp;

For 2D with $A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$:

$$Q = ax_0^2 + 2bx_0x_1 + cx_1^2$$

&nbsp;

*Workout:* Expand $Q$ for $A = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$:

**Solution:**

Using $Q = ax_0^2 + 2bx_0x_1 + cx_1^2$ with $a=3$, $b=1$, $c=2$:

$$Q(\vec{x}) = \begin{bmatrix} x_0 & x_1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = 3x_0^2 + 2x_0x_1 + 2x_1^2$$

---

## Geometric Meaning: Conic Sections

Quadratic forms describe **conic sections**:

| Equation | Shape |
|----------|-------|
| $(\vec{x}-\vec{a})^T \mathbf{I} (\vec{x}-\vec{a}) = r^2$ | Circle |
| $(\vec{x}-\vec{a})^T A (\vec{x}-\vec{a}) = 1$ | Ellipse |

&nbsp;

**Key Insight:** Level sets of $Q$ are ellipses (or hyperbolas) whose axes align with eigenvectors!

&nbsp;



---

## Quadratic Forms in ML

**Where quadratic forms appear:**

1. **Taylor expansion:** $L(\vec{x} + \delta) \approx L(\vec{x}) + \nabla L \cdot \delta + \frac{1}{2}\delta^T H \delta$

2. **Squared error:** $\|\vec{y} - \hat{\vec{y}}\|^2 = (\vec{y} - \hat{\vec{y}})^T(\vec{y} - \hat{\vec{y}})$

3. **Regularization:** $\|\vec{w}\|^2 = \vec{w}^T \vec{w}$

&nbsp;

---

## Extrema on the Unit Sphere

**Question:** For unit vectors $\hat{x}$ (with $\|\hat{x}\| = 1$), what are the max/min of $Q$?

&nbsp;

**Key Trick:** Use spectral decomposition $A = S\Lambda S^T$:

$$Q = \hat{x}^T A \hat{x} = \hat{y}^T \Lambda \hat{y} = \sum_{i=1}^n \lambda_i y_i^2$$

where $\hat{y} = S^T \hat{x}$ is also a unit vector.

&nbsp;

---

## Theorem: Eigenvalues = Extrema

Since $\sum_i y_i^2 = 1$ and $y_i^2 \geq 0$:

$$Q = \sum_i \lambda_i y_i^2$$

is a **weighted average** of eigenvalues.

&nbsp;

**Result:**

- **Maximum:** $Q_{max} = \lambda_{max}$ when $\hat{x} = \vec{e}_1$ (largest eigenvector)
- **Minimum:** $Q_{min} = \lambda_{min}$ when $\hat{x} = \vec{e}_n$ (smallest eigenvector)

&nbsp;

*Workout:* If $\lambda_1 = 5, \lambda_2 = 2$, what are max/min of $Q$ on unit circle?

**Solution:**

By the extrema theorem, on the unit sphere the quadratic form $Q = \sum_i \lambda_i y_i^2$ is maximized when all weight is on the largest eigenvalue and minimized when all weight is on the smallest:

$$Q_{max} = \lambda_1 = 5 \quad \text{(achieved at eigenvector } \vec{e}_1\text{)}$$

$$Q_{min} = \lambda_2 = 2 \quad \text{(achieved at eigenvector } \vec{e}_2\text{)}$$

---

## Positive Definite Matrices

**Definition:** Symmetric $A$ is:

- **Positive Definite (PD):** $\vec{x}^T A \vec{x} > 0$ for all $\vec{x} \neq 0$
- **Positive Semidefinite (PSD):** $\vec{x}^T A \vec{x} \geq 0$ for all $\vec{x}$

&nbsp;

**Eigenvalue Test:**

$$A \text{ is PD} \iff \text{all } \lambda_i > 0$$
$$A \text{ is PSD} \iff \text{all } \lambda_i \geq 0$$

&nbsp;

---

## Why Positive Definiteness Matters

**ML Applications:**

1. **Covariance matrices** are always PSD
2. **Hessian at minimum** must be PD (second derivative test)
3. **Convex losses** have PSD Hessians everywhere

&nbsp;

*Workout:* Is $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ positive definite? (Find eigenvalues)

**Solution:**

Characteristic equation: $\det(A - \lambda I) = (2-\lambda)^2 - 1 = 0$

$$\lambda^2 - 4\lambda + 3 = 0 \implies (\lambda - 3)(\lambda - 1) = 0$$

$$\lambda_1 = 3, \quad \lambda_2 = 1$$

Both eigenvalues are strictly positive, so **$A$ is positive definite**.

---

## Indefinite Matrices: Saddle Points

If $A$ has both positive and negative eigenvalues:

- $A$ is **indefinite**
- Level sets are **hyperbolas**
- Critical point is a **saddle point**

&nbsp;

*Workout:* Classify $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$:

Eigenvalues: $\det(A - \lambda I) = 0$

**Solution:**

$$(1-\lambda)^2 - 4 = 0 \implies \lambda^2 - 2\lambda - 3 = 0 \implies (\lambda - 3)(\lambda + 1) = 0$$

$$\lambda_1 = 3 > 0, \quad \lambda_2 = -1 < 0$$

One positive and one negative eigenvalue → **$A$ is indefinite**. The critical point of $Q(\vec{x}) = \vec{x}^T A \vec{x}$ is a **saddle point**, and the level sets are hyperbolas.

---

## Condition Number

**Definition:** For positive definite $A$:

$$\kappa(A) = \frac{\lambda_{max}}{\lambda_{min}}$$

&nbsp;

**Interpretation:**
- $\kappa \approx 1$: circular level sets → fast convergence
- $\kappa \gg 1$: elongated ellipses → slow convergence (zig-zagging)

&nbsp;

**Ideal:** $\kappa = 1$ (all eigenvalues equal)

&nbsp;

---

## Spectral Norm

**Definition:** Maximum amplification by matrix $A$:

$$\|A\|_2 = \max_{\|\hat{x}\|=1} \|A\hat{x}\|$$

&nbsp;

**Result:** Spectral norm = largest singular value:

$$\|A\|_2 = \sigma_1 = \sqrt{\lambda_{max}(A^T A)}$$

&nbsp;

**ML Use:** Lipschitz constant of linear layer, spectral normalization.

&nbsp;

---

## Computing Spectral Norm

For unit vector $\hat{x}$:

$$\|A\hat{x}\|^2 = (A\hat{x})^T(A\hat{x}) = \hat{x}^T A^T A \hat{x}$$

This is a quadratic form in $A^T A$!

&nbsp;

**Maximum occurs** along eigenvector of $A^T A$ with largest eigenvalue.

&nbsp;

*Workout:* Find $\|A\|_2$ for $A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$:

**Solution:**

$A$ is diagonal, so its singular values equal the absolute values of its diagonal entries: $\sigma_1 = 3$, $\sigma_2 = 2$.

$$\|A\|_2 = \sigma_1 = 3$$

Equivalently: $A^T A = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}$, so $\lambda_{max}(A^T A) = 9$ and $\|A\|_2 = \sqrt{9} = 3$.

---

## Frobenius Norm

**Definition:** "Size" of a matrix (like L2 for vectors):

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}$$

&nbsp;

**Also equals:**

$$\|A\|_F = \sqrt{\sum_i \sigma_i^2} = \sqrt{\text{trace}(A^T A)}$$

&nbsp;

**ML Use:** Regularization, approximation error.

&nbsp;

*Workout:* Compute $\|A\|_F$ for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$:

**Solution:**

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{1^2 + 2^2 + 3^2 + 4^2} = \sqrt{1 + 4 + 9 + 16} = \sqrt{30} \approx 5.477$$

---

## Norm Comparison

**Always:** $\|A\|_2 \leq \|A\|_F$

**Equality when:** $A$ has rank 1.

&nbsp;

| Norm | Formula | Measures |
|------|---------|----------|
| Spectral | $\sigma_1$ | Max stretch |
| Frobenius | $\sqrt{\sum \sigma_i^2}$ | Total "size" |

&nbsp;

---

## Unit Circle Transformation

When $A$ acts on unit circle:

- Circle maps to **ellipse**
- Semi-axes = singular values $\sigma_1, \sigma_2$
- Spectral norm = longest semi-axis

&nbsp;

*Workout:* Sketch how $A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$ transforms unit circle:

**Solution:**

$A$ is diagonal with singular values $\sigma_1 = 2$, $\sigma_2 = 1$.

The unit circle $x_0^2 + x_1^2 = 1$ maps to an **ellipse** with semi-axis length 2 along $x_0$ and semi-axis length 1 along $x_1$:

$$\frac{x_0^2}{4} + x_1^2 = 1$$

The $x_0$ direction is stretched by factor 2, while the $x_1$ direction is unchanged. The spectral norm is $\|A\|_2 = 2$ (the longest semi-axis).

---

## Summary: Key Formulas

| Concept | Formula |
|---------|---------|
| Quadratic Form | $Q = \vec{x}^T A \vec{x}$ |
| Max/Min on sphere | $\lambda_{max}$, $\lambda_{min}$ |
| PD Test | All $\lambda_i > 0$ |
| Condition Number | $\kappa = \lambda_{max}/\lambda_{min}$ |
| Spectral Norm | $\|A\|_2 = \sigma_1$ |
| Frobenius Norm | $\|A\|_F = \sqrt{\sum \sigma_i^2}$ |




