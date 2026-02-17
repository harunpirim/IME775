# IME 775 — Chapter 5 Problem Set
## Probability Distributions in Machine Learning

---

### Part A: Probability Basics

**Problem 1.** A dataset has 200 images: 80 cats, 70 dogs, 50 birds.
(a) What is $P(\text{cat})$?
(b) What is $P(\text{not bird})$?
(c) If we randomly pick 2 images (with replacement), what is $P(\text{both cats})$?

**Solution:**
(a) $P(\text{cat}) = \frac{80}{200} = 0.40$

(b) $P(\text{not bird}) = 1 - P(\text{bird}) = 1 - \frac{50}{200} = 0.75$

(c) With replacement, draws are independent: $P(\text{both cats}) = 0.4 \times 0.4 = 0.16$

---

**Problem 2.** Given the joint probability table:

| | $Y=0$ | $Y=1$ |
|---|:---:|:---:|
| $X=0$ | 0.30 | 0.10 |
| $X=1$ | 0.20 | 0.40 |

(a) Find the marginal distributions $P(X)$ and $P(Y)$.
(b) Are $X$ and $Y$ independent?
(c) Compute $P(Y=1 | X=1)$.

**Solution:**
(a) Marginals:
- $P(X=0) = 0.30 + 0.10 = 0.40$, $P(X=1) = 0.20 + 0.40 = 0.60$
- $P(Y=0) = 0.30 + 0.20 = 0.50$, $P(Y=1) = 0.10 + 0.40 = 0.50$

(b) Check: $P(X=0, Y=0) = 0.30$ vs $P(X=0) \cdot P(Y=0) = 0.40 \times 0.50 = 0.20$.
Since $0.30 \neq 0.20$, $X$ and $Y$ are **not independent**.

(c) $P(Y=1 | X=1) = \frac{P(X=1, Y=1)}{P(X=1)} = \frac{0.40}{0.60} = \frac{2}{3} \approx 0.667$

---

### Part B: Random Variables and Expectation

**Problem 3.** A random variable $X$ has PMF:

| $x$ | 1 | 2 | 3 | 4 |
|---|:---:|:---:|:---:|:---:|
| $P(X=x)$ | 0.1 | 0.3 | 0.4 | 0.2 |

(a) Verify this is a valid PMF.
(b) Compute $E[X]$.
(c) Compute $\text{Var}(X)$.
(d) Compute $E[3X + 2]$.

**Solution:**
(a) $\sum p(x) = 0.1 + 0.3 + 0.4 + 0.2 = 1.0$ ✓ and all $p(x) \geq 0$ ✓

(b) $E[X] = 1(0.1) + 2(0.3) + 3(0.4) + 4(0.2) = 0.1 + 0.6 + 1.2 + 0.8 = 2.7$

(c) $E[X^2] = 1(0.1) + 4(0.3) + 9(0.4) + 16(0.2) = 0.1 + 1.2 + 3.6 + 3.2 = 8.1$
$\text{Var}(X) = E[X^2] - (E[X])^2 = 8.1 - 7.29 = 0.81$

(d) $E[3X + 2] = 3E[X] + 2 = 3(2.7) + 2 = 10.1$

---

**Problem 4.** A continuous random variable has PDF $f(x) = 3x^2$ for $x \in [0,1]$, and $0$ otherwise.

(a) Verify $\int_0^1 f(x)\,dx = 1$.
(b) Compute $P(X > 0.5)$.
(c) Compute $E[X]$ and $\text{Var}(X)$.

**Solution:**
(a) $\int_0^1 3x^2\,dx = [x^3]_0^1 = 1$ ✓

(b) $P(X > 0.5) = \int_{0.5}^1 3x^2\,dx = [x^3]_{0.5}^1 = 1 - 0.125 = 0.875$

(c) $E[X] = \int_0^1 x \cdot 3x^2\,dx = \int_0^1 3x^3\,dx = \left[\frac{3x^4}{4}\right]_0^1 = \frac{3}{4} = 0.75$

$E[X^2] = \int_0^1 x^2 \cdot 3x^2\,dx = \int_0^1 3x^4\,dx = \left[\frac{3x^5}{5}\right]_0^1 = \frac{3}{5} = 0.6$

$\text{Var}(X) = 0.6 - 0.75^2 = 0.6 - 0.5625 = 0.0375$

---

### Part C: Covariance Matrix

**Problem 5.** Given data points: $(1,2)$, $(3,4)$, $(5,6)$.

(a) Compute the mean vector $\vec{\mu}$.
(b) Compute the covariance matrix $\Sigma$.
(c) What do the diagonal entries tell you? The off-diagonal?

**Solution:**
(a) $\vec{\mu} = \left(\frac{1+3+5}{3}, \frac{2+4+6}{3}\right) = (3, 4)$

(b) Centered data: $(-2,-2)$, $(0,0)$, $(2,2)$

$\Sigma = \frac{1}{3}\begin{pmatrix} (-2)^2 + 0 + 2^2 & (-2)(-2) + 0 + (2)(2) \\ (-2)(-2) + 0 + (2)(2) & (-2)^2 + 0 + 2^2 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 8 & 8 \\ 8 & 8 \end{pmatrix} = \begin{pmatrix} 8/3 & 8/3 \\ 8/3 & 8/3 \end{pmatrix}$

(c) Diagonal entries: $\text{Var}(X_1) = \text{Var}(X_2) = 8/3 \approx 2.67$. Both features have equal spread.
Off-diagonal: $\text{Cov}(X_1, X_2) = 8/3$. Equal to the variances → perfect positive linear correlation. The data lies on a line.

---

**Problem 6.** A covariance matrix is $\Sigma = \begin{pmatrix} 5 & -2 \\ -2 & 3 \end{pmatrix}$.

(a) Is $\Sigma$ positive definite? (Check eigenvalues.)
(b) What does the negative off-diagonal entry mean?
(c) Sketch the approximate shape of the 1-$\sigma$ ellipse.

**Solution:**
(a) Eigenvalues: $\lambda^2 - 8\lambda + 11 = 0 \Rightarrow \lambda = \frac{8 \pm \sqrt{64-44}}{2} = \frac{8 \pm \sqrt{20}}{2}$
$\lambda_1 \approx 6.24$, $\lambda_2 \approx 1.76$. Both positive → **positive definite** ✓

(b) $\text{Cov}(X_1, X_2) = -2 < 0$ means the variables are **negatively correlated**: when $X_1$ increases, $X_2$ tends to decrease.

(c) The ellipse has semi-axes $\sqrt{6.24} \approx 2.50$ and $\sqrt{1.76} \approx 1.33$, rotated so the major axis tilts toward the direction where $X_1$ increases and $X_2$ decreases (upper-left to lower-right trend).

---

### Part D: Gaussian Distribution

**Problem 7.** For $X \sim \mathcal{N}(10, 25)$:

(a) What is $P(5 \leq X \leq 15)$? (Use the 68-95-99.7 rule.)
(b) What is $P(X > 20)$?
(c) If we standardize $Z = \frac{X - \mu}{\sigma}$, what is the distribution of $Z$?

**Solution:**
(a) $\sigma = \sqrt{25} = 5$. The interval $[5, 15] = [\mu - \sigma, \mu + \sigma]$.
By the 68-95-99.7 rule: $P(5 \leq X \leq 15) \approx 68\%$

(b) $X = 20 = \mu + 2\sigma$. $P(X > \mu + 2\sigma) \approx \frac{1 - 0.95}{2} = 2.5\%$

(c) $Z \sim \mathcal{N}(0, 1)$ — the **standard normal** distribution.

---

**Problem 8.** For the multivariate Gaussian with $\vec{\mu} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, $\Sigma = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$:

(a) Are $X_1$ and $X_2$ independent?
(b) Describe the shape of the constant-probability contours.
(c) What is the Mahalanobis distance of the point $(2, 1)$ from the mean?

**Solution:**
(a) Yes! Since $\Sigma$ is diagonal, $\text{Cov}(X_1, X_2) = 0$. For Gaussians, zero covariance implies independence.

(b) Axis-aligned ellipses with semi-axis lengths proportional to $\sqrt{4} = 2$ along $x_1$ and $\sqrt{1} = 1$ along $x_2$.

(c) Mahalanobis distance:
$d = \sqrt{(\vec{x} - \vec{\mu})^T \Sigma^{-1} (\vec{x} - \vec{\mu})} = \sqrt{(2, 1) \begin{pmatrix} 1/4 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix}}$
$= \sqrt{\frac{4}{4} + \frac{1}{1}} = \sqrt{1 + 1} = \sqrt{2} \approx 1.41$

---

**Problem 9.** Given three 2D Gaussian distributions:
- $\mathcal{N}_A$: $\mu = (0,0)$, $\Sigma = I$
- $\mathcal{N}_B$: $\mu = (0,0)$, $\Sigma = \begin{pmatrix} 3 & 0 \\ 0 & 0.5 \end{pmatrix}$
- $\mathcal{N}_C$: $\mu = (0,0)$, $\Sigma = \begin{pmatrix} 2 & 1.5 \\ 1.5 & 2 \end{pmatrix}$

Match each to its contour description: (i) circle, (ii) axis-aligned ellipse, (iii) rotated ellipse.

**Solution:**
- $\mathcal{N}_A \to$ (i) circle: $\Sigma = I$ means equal variance in all directions
- $\mathcal{N}_B \to$ (ii) axis-aligned ellipse: diagonal $\Sigma$ with unequal entries
- $\mathcal{N}_C \to$ (iii) rotated ellipse: non-zero off-diagonal entries → correlation → rotation

---

### Part E: Bernoulli, Categorical, Multinomial

**Problem 10.** A coin has $P(\text{heads}) = 0.6$.

(a) Write the Bernoulli PMF for one flip ($X=1$ means heads).
(b) Compute $E[X]$ and $\text{Var}(X)$.
(c) For 10 flips, what is the expected number of heads?
(d) What is $P(\text{exactly 7 heads in 10 flips})$?

**Solution:**
(a) $P(X = x) = 0.6^x \cdot 0.4^{1-x}$ for $x \in \{0, 1\}$

(b) $E[X] = 0.6$, $\text{Var}(X) = 0.6 \times 0.4 = 0.24$

(c) $E[\text{heads}] = n\theta = 10 \times 0.6 = 6$

(d) $P(X = 7) = \binom{10}{7} 0.6^7 \cdot 0.4^3 = 120 \times 0.0280 \times 0.064 = 120 \times 0.001792 \approx 0.215$

---

**Problem 11.** A text classifier assigns probabilities: $P(\text{sports}) = 0.6$, $P(\text{politics}) = 0.3$, $P(\text{science}) = 0.1$.

(a) Write this as a Categorical distribution with one-hot encoding.
(b) In a corpus of 500 documents, how many do you expect in each category?
(c) What is the variance of the sports document count?

**Solution:**
(a) $\vec{\theta} = (0.6, 0.3, 0.1)$. For a single document, $P(\vec{x} = \vec{e}_k) = \theta_k$.

(b) Expected counts (Multinomial): $E[m_k] = n\theta_k$
- Sports: $500 \times 0.6 = 300$
- Politics: $500 \times 0.3 = 150$
- Science: $500 \times 0.1 = 50$

(c) $\text{Var}(m_\text{sports}) = n\theta(1-\theta) = 500 \times 0.6 \times 0.4 = 120$, so $\sigma \approx 10.95$ documents.

---

**Problem 12.** A vocabulary has 3 words: {"the", "cat", "sat"} with probabilities $\vec{\theta} = (0.5, 0.3, 0.2)$.

(a) For a 10-word document, what is the expected word count vector?
(b) What is $P(\text{document} = [5, 3, 2])$? (Use the Multinomial PMF.)

**Solution:**
(a) $E[\vec{m}] = n\vec{\theta} = 10 \times (0.5, 0.3, 0.2) = (5, 3, 2)$

(b) $P(\vec{m} = [5,3,2]) = \frac{10!}{5! \cdot 3! \cdot 2!} \cdot 0.5^5 \cdot 0.3^3 \cdot 0.2^2$
$= \frac{3628800}{120 \cdot 6 \cdot 2} \cdot 0.03125 \cdot 0.027 \cdot 0.04$
$= 2520 \times 0.00003375$
$= 0.08505$

---

### Part F: Connections to ML

**Problem 13.** In logistic regression, the output is $P(y = 1 | \vec{x}) = \sigma(\vec{w}^T\vec{x} + b)$ where $\sigma(z) = \frac{1}{1+e^{-z}}$.

(a) What distribution does this model?
(b) If $\vec{w}^T\vec{x} + b = 2$, what is $P(y=1)$?
(c) What happens as $\vec{w}^T\vec{x} + b \to +\infty$?

**Solution:**
(a) **Bernoulli** distribution with parameter $\theta = \sigma(\vec{w}^T\vec{x} + b)$

(b) $P(y=1) = \sigma(2) = \frac{1}{1+e^{-2}} = \frac{1}{1+0.135} \approx 0.881$

(c) $\sigma(z) \to 1$ as $z \to +\infty$, so the model becomes completely confident that $y = 1$.

---

**Problem 14.** The softmax function converts logits $\vec{z}$ to probabilities:

$$\text{softmax}_k(\vec{z}) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

(a) What distribution does softmax produce?
(b) For $\vec{z} = (2, 1, 0)$, compute the softmax probabilities.
(c) Verify $\sum_k \text{softmax}_k = 1$.

**Solution:**
(a) **Categorical** distribution with $\theta_k = \text{softmax}_k(\vec{z})$

(b) $e^2 \approx 7.389$, $e^1 \approx 2.718$, $e^0 = 1.0$. Sum $= 11.107$.
- $\text{softmax}_1 = 7.389 / 11.107 \approx 0.665$
- $\text{softmax}_2 = 2.718 / 11.107 \approx 0.245$
- $\text{softmax}_3 = 1.0 / 11.107 \approx 0.090$

(c) $0.665 + 0.245 + 0.090 = 1.0$ ✓

---

### Part G: Conceptual Questions

**Problem 15.** True or False (with brief justification):

(a) A PDF $f(x)$ can have values greater than 1.
(b) If $\text{Cov}(X, Y) = 0$, then $X$ and $Y$ are independent.
(c) The Multinomial distribution is to the Categorical as the Binomial is to the Bernoulli.
(d) The covariance matrix is always positive semi-definite.

**Solution:**
(a) **True.** For example, $f(x) = 3$ on $[0, 1/3]$. Only areas (integrals) must be ≤ 1, not the density values.

(b) **False.** Zero covariance means no *linear* relationship. There can be nonlinear dependencies. However, for Gaussian random variables, zero covariance *does* imply independence.

(c) **True.** Multinomial counts outcomes over $n$ trials with $K$ categories, just as Binomial counts successes over $n$ trials with 2 outcomes.

(d) **True.** For any vector $\vec{v}$: $\vec{v}^T\Sigma\vec{v} = \vec{v}^T E[(\vec{X}-\vec{\mu})(\vec{X}-\vec{\mu})^T]\vec{v} = E[(\vec{v}^T(\vec{X}-\vec{\mu}))^2] \geq 0$.

---

**Problem 16.** Explain the relationship chain:

Covariance Matrix $\to$ Eigenvalues/Eigenvectors $\to$ PCA $\to$ Gaussian Contours

**Solution:**
1. The **covariance matrix** $\Sigma$ captures how data spreads in each direction.
2. **Eigendecomposition** of $\Sigma$ gives eigenvectors (principal directions) and eigenvalues (variance in those directions).
3. **PCA** projects data onto the top eigenvectors, reducing dimensionality while preserving maximum variance.
4. **Gaussian contours** are ellipses defined by $\Sigma$: axes aligned with eigenvectors, lengths proportional to $\sqrt{\lambda_i}$. Same eigenvectors, same geometry!
