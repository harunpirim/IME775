# IME 775 — Lecture 6
## PCA, SVD, and Document Retrieval

---

## 1. The Dimensionality Reduction Problem

**Observation:** Real data often lies near a lower-dimensional subspace.

**Example:** Document vectors cluster around "topics" (gun-violence, sports, etc.)

**Goal:** Project onto subspace to:
- Reduce storage
- Remove noise
- Reveal structure

---

## 2. Measuring Spread: 1D Variance

For data $\{x^{(0)}, x^{(1)}, \ldots, x^{(n)}\}$:

**Mean:** $\mu = \frac{1}{n}\sum_{i} x^{(i)}$

**Variance:** $\sigma^2 = \frac{1}{n}\sum_{i} (x^{(i)} - \mu)^2$

&nbsp;

*Workout:* Compute mean and variance for $\{1, 2, 3, 4, 5\}$:

&nbsp;

&nbsp;

&nbsp;

---

## 3. Multidimensional: Mean Vector

For data points $\vec{x}^{(0)}, \vec{x}^{(1)}, \ldots, \vec{x}^{(n)}$:

$$\vec{\mu} = \frac{1}{n}\sum_{i=0}^{n} \vec{x}^{(i)}$$

&nbsp;

Mean is now a **vector** (centroid of data).

&nbsp;

---

## 4. Covariance Matrix

**Definition:**

$$C = \frac{1}{n}\sum_{i=0}^{n} (\vec{x}^{(i)} - \vec{\mu})(\vec{x}^{(i)} - \vec{\mu})^T$$

&nbsp;

For centered data ($\vec{\mu} = 0$):

$$C = \frac{1}{n} X^T X$$

&nbsp;

**Structure:**
- Diagonal: variance along each axis
- Off-diagonal: covariance between axes

---

## 5. Covariance Matrix Example

$$C = \begin{bmatrix} \sigma_{00} & \sigma_{01} \\ \sigma_{10} & \sigma_{11} \end{bmatrix}$$

&nbsp;

- $\sigma_{00}$: variance in $x_0$ direction
- $\sigma_{11}$: variance in $x_1$ direction
- $\sigma_{01} = \sigma_{10}$: covariance (how $x_0, x_1$ vary together)

&nbsp;

*Workout:* Is covariance matrix always symmetric? Why?

&nbsp;

&nbsp;

---

## 6. Variance Along a Direction

**Question:** What's the variance along direction $\hat{l}$?

&nbsp;

**Projection:** Component of $\vec{x}^{(i)}$ along $\hat{l}$ is $\hat{l}^T \vec{x}^{(i)}$

&nbsp;

**Variance along $\hat{l}$:**

$$\sigma^2_{\hat{l}} = \hat{l}^T C \hat{l}$$

&nbsp;

This is a **quadratic form**!

---

## 7. PCA: Core Insight

**From Lecture 5:** Quadratic form $\hat{l}^T C \hat{l}$ is maximized when $\hat{l}$ is the eigenvector of $C$ with largest eigenvalue.

&nbsp;

**Therefore:** Direction of maximum variance = eigenvector of covariance matrix!

&nbsp;

---

## 8. Principal Components

**First Principal Component (PC1):**
- Direction: eigenvector of $C$ for $\lambda_1$ (largest)
- Variance: $\lambda_1$

**Second Principal Component (PC2):**
- Direction: eigenvector for $\lambda_2$
- Orthogonal to PC1 (since $C$ symmetric)

&nbsp;

**Key Property:** PCs are **mutually orthogonal**!

---

## 9. Variance Explained

**Total variance:** $\text{Total} = \sum_{i} \lambda_i = \text{trace}(C)$

**Variance explained by PC$_i$:**

$$\frac{\lambda_i}{\sum_j \lambda_j} \times 100\%$$

&nbsp;

*Workout:* If $\lambda_1 = 8, \lambda_2 = 2$, what % does PC1 explain?

&nbsp;

&nbsp;

---

## 10. PCA Algorithm

**Input:** Data matrix $X$ (rows = samples)

1. **Center:** $X_c = X - \bar{x}$

2. **Covariance:** $C = \frac{1}{n}X_c^T X_c$

3. **Eigendecomposition:** $C = Q\Lambda Q^T$

4. **Project:** $Z = X_c Q_k$ (keep top $k$ eigenvectors)

&nbsp;

**Reconstruction:** $\hat{X} = Z Q_k^T + \bar{x}$

---

## 11. Reconstruction Error

$$\text{Error} = \sum_{i=k+1}^{n} \lambda_i$$

(Sum of discarded eigenvalues)

&nbsp;

*Workout:* With $\lambda_1 = 10, \lambda_2 = 5, \lambda_3 = 1$, what's error keeping 2 PCs?

&nbsp;

&nbsp;

---

## 12. PCA Limitations

**PCA assumes linear patterns!**

&nbsp;

If data follows a curve:
- PCA finds best linear approximation
- Misses the true structure

&nbsp;

*Workout:* Sketch 2D data where PCA fails:

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## 13. Singular Value Decomposition

**Theorem:** Any matrix $A \in \mathbb{R}^{m \times n}$ can be written:

$$A = U \Sigma V^T$$

&nbsp;

- $U$: $m \times m$ orthogonal (left singular vectors)
- $\Sigma$: $m \times n$ diagonal (singular values)
- $V$: $n \times n$ orthogonal (right singular vectors)

---

## 14. SVD Components

$$\sigma_i = \sqrt{\lambda_i(A^T A)}$$

&nbsp;

- $V$: eigenvectors of $A^T A$
- $U$: eigenvectors of $AA^T$
- Singular values: always non-negative

&nbsp;

**Key:** SVD works for ANY matrix (not just square)!

---

## 15. SVD and PCA Connection

For centered data matrix $X$:

$$X^T X = C \text{ (covariance)}$$

&nbsp;

**SVD of $X$ gives:**
- Columns of $V$ = principal vectors
- $\Sigma^2 / n$ = principal values

&nbsp;

**SVD is an efficient way to compute PCA!**

---

## 16. SVD for Solving Linear Systems

For $A\vec{x} = \vec{b}$, using $A = U\Sigma V^T$:

1. $U\vec{y}_1 = \vec{b}$ → $\vec{y}_1 = U^T\vec{b}$
2. $\Sigma\vec{y}_2 = \vec{y}_1$ → $\vec{y}_2 = \Sigma^{-1}\vec{y}_1$
3. $V^T\vec{x} = \vec{y}_2$ → $\vec{x} = V\vec{y}_2$

&nbsp;

**Advantage:** No explicit matrix inversion needed!

---

## 17. Matrix Rank via SVD

**Rank** = number of nonzero singular values

&nbsp;

**Degenerate system:** Some $\sigma_i = 0$
- Rows linearly dependent
- det$(A) = 0$
- No unique solution

&nbsp;

---

## 18. Low-Rank Approximation

**Best rank-$r$ approximation:**

$$A_r = \sum_{i=1}^{r} \sigma_i \vec{u}_i \vec{v}_i^T$$

&nbsp;

**Error:** $\|A - A_r\|_F = \sqrt{\sum_{i=r+1}^{p} \sigma_i^2}$

&nbsp;

*Workout:* If $\sigma_1 = 10, \sigma_2 = 3, \sigma_3 = 1$, what's rank-2 approx error?

&nbsp;

&nbsp;

---

## 19. Document Retrieval: TF-IDF

**Term Frequency (TF):** Word count in document

**Inverse Document Frequency (IDF):** Down-weight common words

$$\text{TF-IDF} = \text{TF} \times \log\frac{N}{\text{docs with term}}$$

&nbsp;

**Cosine Similarity:**

$$\cos(\vec{a}, \vec{b}) = \frac{\vec{a}^T\vec{b}}{\|\vec{a}\|\|\vec{b}\|}$$

---

## 20. Problem with Direct Comparison

**d5:** "Guns were used in robbery" (contains "gun")
**d6:** "Acts of violence" (contains "violence")

&nbsp;

**Direct cosine similarity:** 0 (no shared terms)

But intuitively they're related!

&nbsp;

---

## 21. Latent Semantic Analysis (LSA)

**Idea:** Find "topics" = combinations of terms that co-occur

&nbsp;

**Algorithm:**
1. Create doc-term matrix $X$
2. SVD: $X = U\Sigma V^T$
3. Columns of $V$ = topics
4. Project docs to topic space
5. Compare in topic space

---

## 22. LSA Example

"Gun" and "violence" co-occur in many documents.

→ They form a **topic** (linear combination)

→ Documents about "gun" OR "violence" both score high on this topic

→ **Latent similarity** revealed!

&nbsp;

---

## 23. LSA Implementation

```python
# Doc-term matrix X
U, S, V_t = torch.linalg.svd(X)
V = V_t.T

# Keep top k topics
V_topics = V[:, :k]

# Project to topic space
X_topics = X @ V_topics

# Similarity in topic space
sim = cosine_similarity(X_topics[i], X_topics[j])
```

---

## Summary: Key Formulas

| Concept | Formula |
|---------|---------|
| Covariance | $C = \frac{1}{n}X^TX$ |
| Variance along $\hat{l}$ | $\hat{l}^T C \hat{l}$ |
| PCA direction | Eigenvector of $C$ |
| Variance explained | $\lambda_i / \sum \lambda_j$ |
| SVD | $A = U\Sigma V^T$ |
| Low-rank error | $\sqrt{\sum_{i>r} \sigma_i^2}$ |
| Cosine similarity | $\frac{\vec{a}^T\vec{b}}{\|\vec{a}\|\|\vec{b}\|}$ |

---

## Exercises for iPad Workout

1. Compute covariance matrix for points $(0,0), (1,1), (2,2)$

2. If $C = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$, find direction of max variance

3. Data has $\lambda_1=50, \lambda_2=30, \lambda_3=15, \lambda_4=5$. How many PCs for 90%?

4. Given $\sigma_1=8, \sigma_2=4, \sigma_3=2$, compute $\|A\|_F$

5. Why does LSA find similarity between docs with no shared terms?

---

*IME 775 — Mathematical Foundations of Deep Learning*
