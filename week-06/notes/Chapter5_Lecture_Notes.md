# IME 775 — Lecture 8
## Probability Foundations for Machine Learning

---

## 1. Why Probability in ML?

Machine learning models must handle **uncertainty**:
- Noisy sensor readings
- Incomplete data
- Inherent randomness in natural processes

**Key idea:** We model data as outcomes of random experiments, then learn patterns from observed outcomes.

**ML Connection:** Training = estimating probability distributions from data

---

## 2. Probability Basics

### Sample Space and Events

**Sample space** $\Omega$: set of all possible outcomes

**Event** $E \subseteq \Omega$: a subset of outcomes

**Examples:**

| Experiment | Sample Space $\Omega$ | Event $E$ |
|---|---|---|
| Coin flip | $\{H, T\}$ | $E = \{H\}$ |
| Die roll | $\{1,2,3,4,5,6\}$ | $E = \{2,4,6\}$ (even) |
| Image class | $\{$cat, dog, bird$\}$ | $E = \{$cat$\}$ |

### Probability of an Event

$$P(E) = \frac{|E|}{|\Omega|} \quad \text{(equally likely outcomes)}$$

**Properties:**
- $0 \leq P(E) \leq 1$
- $P(\Omega) = 1$
- $P(\emptyset) = 0$
- $P(\bar{E}) = 1 - P(E)$

&nbsp;

*Workout:* A bag has 3 red, 2 blue, 5 green balls. What is $P(\text{red or blue})$?

**Solution:**
$P(\text{red or blue}) = \frac{3 + 2}{3 + 2 + 5} = \frac{5}{10} = 0.5$

---

## 3. Random Variables

A **random variable** $X$ is a function that maps outcomes to numbers:

$$X: \Omega \to \mathbb{R}$$

### Discrete Random Variables

Takes countable values: $X \in \{x_1, x_2, \ldots\}$

**Probability Mass Function (PMF):**
$$p(X = x_i) = P(\{\omega \in \Omega : X(\omega) = x_i\})$$

**Example:** $X$ = number on a fair die

$$p(X = k) = \frac{1}{6}, \quad k \in \{1,2,3,4,5,6\}$$

### Continuous Random Variables

Takes values in an interval: $X \in \mathbb{R}$

**Probability Density Function (PDF):** $f(x)$ such that

$$P(a \leq X \leq b) = \int_a^b f(x)\,dx$$

**Key property:** $f(x) \geq 0$ but $f(x)$ can exceed 1! Only areas represent probabilities.

### Cumulative Distribution Function (CDF)

$$F(x) = P(X \leq x)$$

For continuous: $F(x) = \int_{-\infty}^{x} f(t)\,dt$

For discrete: $F(x) = \sum_{x_i \leq x} p(X = x_i)$

&nbsp;

*Workout:* If $f(x) = 2x$ for $x \in [0,1]$ and 0 otherwise, find $P(0.5 \leq X \leq 1)$.

**Solution:**
$$P(0.5 \leq X \leq 1) = \int_{0.5}^{1} 2x\,dx = \left[x^2\right]_{0.5}^{1} = 1 - 0.25 = 0.75$$

---

## 4. Joint and Marginal Probability

### Joint Probability

For two random variables $X$ and $Y$:

$$P(X = x_i, Y = y_j) = P(X = x_i \text{ and } Y = y_j)$$

### Marginal Probability (Sum Rule)

$$P(X = x_i) = \sum_j P(X = x_i, Y = y_j)$$

**Intuition:** "Sum out" the variable you don't care about.

**Example:** Classify images by (animal, color):

| | Brown | White | Black |
|---|:---:|:---:|:---:|
| Cat | 0.10 | 0.15 | 0.05 |
| Dog | 0.20 | 0.10 | 0.10 |
| Bird | 0.05 | 0.10 | 0.15 |

$P(\text{Cat}) = 0.10 + 0.15 + 0.05 = 0.30$

$P(\text{Brown}) = 0.10 + 0.20 + 0.05 = 0.35$

### Product Rule

$$P(X, Y) = P(X | Y) \cdot P(Y) = P(Y | X) \cdot P(X)$$

### Independence

$X$ and $Y$ are independent iff:

$$P(X, Y) = P(X) \cdot P(Y)$$

&nbsp;

*Workout:* From the table above, are "Cat" and "Brown" independent?

**Solution:**
- $P(\text{Cat, Brown}) = 0.10$
- $P(\text{Cat}) \cdot P(\text{Brown}) = 0.30 \times 0.35 = 0.105$
- Since $0.10 \neq 0.105$, they are **not independent**.

---

## 5. Sampling from Distributions

**Sampling** = drawing values according to a distribution.

Given PMF $p(X = x_i)$, sampling produces values where each $x_i$ appears with frequency $\approx p(X = x_i)$.

### Why Sampling Matters in ML

- **Training data** is a sample from the true data distribution
- **Mini-batch SGD** samples subsets for gradient estimation
- **Monte Carlo methods** estimate integrals via sampling
- **Data augmentation** generates new samples

### Empirical Distribution

Given $N$ samples $\{x^{(1)}, \ldots, x^{(N)}\}$:

$$\hat{p}(X = x_i) = \frac{\text{count of } x_i}{N}$$

As $N \to \infty$, $\hat{p} \to p$ (Law of Large Numbers).

---

## 6. Expected Value (Mean)

### Discrete Case

$$E[X] = \sum_i x_i \cdot p(X = x_i)$$

### Continuous Case

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx$$

**Properties:**
- $E[aX + b] = aE[X] + b$ (linearity)
- $E[X + Y] = E[X] + E[Y]$ (always, even if dependent)
- $E[XY] = E[X]E[Y]$ (only if independent)

&nbsp;

*Workout:* A fair die: $E[X] = ?$

**Solution:**
$$E[X] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = \frac{21}{6} = 3.5$$

---

## 7. Variance and Standard Deviation

### Variance

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

### Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if independent)

&nbsp;

*Workout:* For $X \sim \text{Uniform}\{1,2,3,4,5,6\}$, compute $\text{Var}(X)$.

**Solution:**
- $E[X] = 3.5$ (from above)
- $E[X^2] = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.17$
- $\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - \frac{49}{4} = \frac{182 - 147}{12} = \frac{35}{12} \approx 2.92$

---

## 8. Multivariate Expected Value

For a random vector $\vec{X} = (X_1, X_2, \ldots, X_d)^T$:

$$E[\vec{X}] = \begin{pmatrix} E[X_1] \\ E[X_2] \\ \vdots \\ E[X_d] \end{pmatrix} = \vec{\mu}$$

This is just the **mean vector** — same concept as in PCA!

---

## 9. Covariance and Covariance Matrix

### Covariance (2 variables)

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

- $\text{Cov}(X, Y) > 0$: X and Y increase together
- $\text{Cov}(X, Y) < 0$: one increases as other decreases
- $\text{Cov}(X, Y) = 0$: no linear relationship

### Covariance Matrix

For $\vec{X} = (X_1, \ldots, X_d)^T$:

$$\Sigma = E[(\vec{X} - \vec{\mu})(\vec{X} - \vec{\mu})^T]$$

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

**Properties:**
- $\Sigma$ is **symmetric**: $\Sigma_{ij} = \Sigma_{ji}$
- $\Sigma$ is **positive semi-definite**: $\vec{v}^T \Sigma \vec{v} \geq 0$
- Diagonal entries = variances: $\Sigma_{ii} = \text{Var}(X_i)$

**Connection to PCA:** The covariance matrix $\Sigma$ is exactly the matrix whose eigenvectors are the principal components!

&nbsp;

*Workout:* Given $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$, what is $\text{Var}(X_1)$, $\text{Var}(X_2)$, and $\text{Cov}(X_1, X_2)$?

**Solution:**
- $\text{Var}(X_1) = \Sigma_{11} = 4$
- $\text{Var}(X_2) = \Sigma_{22} = 3$
- $\text{Cov}(X_1, X_2) = \Sigma_{12} = 2$ (positive → variables increase together)

---

## Key Takeaways

1. **Random variables** map outcomes to numbers; described by PMF (discrete) or PDF (continuous)
2. **Joint probability** captures relationships; **marginal** is obtained by summing out variables
3. **Expected value** is the "average" outcome; **variance** measures spread
4. **Covariance matrix** captures all pairwise relationships and connects directly to PCA
5. These concepts are the language of machine learning — losses, likelihoods, and Bayesian methods all build on them

---

## PyTorch Connection

```python
import torch

# Sampling from distributions
uniform = torch.distributions.Uniform(0, 1)
samples = uniform.sample((1000,))

# Mean and variance
print(f"Mean: {samples.mean():.4f}")     # ≈ 0.5
print(f"Var:  {samples.var():.4f}")       # ≈ 1/12 ≈ 0.0833

# Covariance matrix from data
data = torch.randn(100, 3)  # 100 samples, 3 features
cov_matrix = torch.cov(data.T)
print(f"Covariance matrix:\n{cov_matrix}")
```
