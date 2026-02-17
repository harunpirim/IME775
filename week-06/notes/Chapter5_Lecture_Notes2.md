# IME 775 — Lecture 9
## Gaussian and Famous Distributions in ML

---

## 1. The Gaussian (Normal) Distribution

The most important distribution in machine learning.

### 1D Gaussian

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Parameters:**
- $\mu$: mean (center of the bell curve)
- $\sigma^2$: variance (width of the bell curve)
- $\sigma$: standard deviation

**Properties:**
- Symmetric about $\mu$
- 68-95-99.7 rule: 68% within $\pm 1\sigma$, 95% within $\pm 2\sigma$, 99.7% within $\pm 3\sigma$
- Peak value: $\frac{1}{\sigma\sqrt{2\pi}}$ at $x = \mu$

**Why Gaussian is everywhere:**
- Central Limit Theorem: sum of many independent random variables → Gaussian
- Maximum entropy distribution for given mean and variance
- Mathematically convenient (closed-form for many operations)

&nbsp;

*Workout:* If $X \sim \mathcal{N}(5, 4)$, what are $\mu$ and $\sigma$? What is $P(3 \leq X \leq 7)$?

**Solution:**
- $\mu = 5$, $\sigma^2 = 4$, so $\sigma = 2$
- $P(3 \leq X \leq 7) = P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$ (by the 68-95-99.7 rule)

---

## 2. Multivariate Gaussian

For $\vec{x} \in \mathbb{R}^d$:

$$\mathcal{N}(\vec{x} \mid \vec{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\vec{x}-\vec{\mu})^T \Sigma^{-1} (\vec{x}-\vec{\mu})\right)$$

**Parameters:**
- $\vec{\mu} \in \mathbb{R}^d$: mean vector
- $\Sigma \in \mathbb{R}^{d \times d}$: covariance matrix (symmetric, positive definite)

### Understanding the Exponent

The expression $(\vec{x}-\vec{\mu})^T \Sigma^{-1} (\vec{x}-\vec{\mu})$ is a **quadratic form** — the Mahalanobis distance!

- Level sets (constant probability) are **ellipses** in 2D
- Axes aligned with **eigenvectors** of $\Sigma$
- Axis lengths proportional to **square roots of eigenvalues**

### Special Cases

| Covariance | Shape | Formula |
|---|---|---|
| $\Sigma = \sigma^2 I$ | Circle (isotropic) | Equal variance in all directions |
| $\Sigma$ diagonal | Axis-aligned ellipse | Independent features |
| $\Sigma$ general | Rotated ellipse | Correlated features |

&nbsp;

*Workout:* For $\Sigma = \begin{pmatrix} 9 & 0 \\ 0 & 1 \end{pmatrix}$, describe the shape of the level sets.

**Solution:**
- Eigenvalues: $\lambda_1 = 9$, $\lambda_2 = 1$
- Eigenvectors: along $x_1$ and $x_2$ axes (diagonal matrix)
- Shape: **axis-aligned ellipse**, stretched along $x_1$ (semi-axis $\sqrt{9} = 3$) and short along $x_2$ (semi-axis $\sqrt{1} = 1$)
- Aspect ratio 3:1

---

## 3. Connection to PCA and Quadratic Forms

**Recall from Chapter 4:**

The quadratic form $Q(\vec{x}) = \vec{x}^T A \vec{x}$ defines ellipses when $A$ is positive definite.

**Now:** The multivariate Gaussian exponent is exactly:

$$Q(\vec{x}) = (\vec{x} - \vec{\mu})^T \Sigma^{-1} (\vec{x} - \vec{\mu})$$

This means:
- **PCA** finds the eigenvectors of $\Sigma$ → principal directions of data spread
- **Gaussian** uses $\Sigma^{-1}$ → same eigenvectors, reciprocal eigenvalues
- Fitting a Gaussian = estimating $\vec{\mu}$ and $\Sigma$ from data

---

## 4. Bernoulli Distribution

Models a single **binary** outcome (coin flip, yes/no, 0/1).

$$\text{Ber}(x \mid \theta) = \theta^x (1 - \theta)^{1-x}, \quad x \in \{0, 1\}$$

**Parameters:** $\theta \in [0,1]$ = probability of success

**Statistics:**
- $E[X] = \theta$
- $\text{Var}(X) = \theta(1 - \theta)$

**ML Applications:**
- Binary classification output (spam / not spam)
- Logistic regression: $P(y=1 | \vec{x}) = \sigma(\vec{w}^T\vec{x} + b)$ where $\sigma$ is the sigmoid function
- Each pixel in binary image generation

&nbsp;

*Workout:* If a spam filter has $P(\text{spam}) = 0.3$, what are $E[X]$ and $\text{Var}(X)$?

**Solution:**
- $E[X] = \theta = 0.3$
- $\text{Var}(X) = 0.3 \times 0.7 = 0.21$

---

## 5. Binomial Distribution

Sum of $n$ independent Bernoulli trials:

$$\text{Bin}(k \mid n, \theta) = \binom{n}{k} \theta^k (1 - \theta)^{n-k}$$

**Parameters:** $n$ = number of trials, $\theta$ = success probability

**Statistics:**
- $E[X] = n\theta$
- $\text{Var}(X) = n\theta(1-\theta)$

**Connection:** If $X_i \sim \text{Ber}(\theta)$ independently, then $\sum_{i=1}^n X_i \sim \text{Bin}(n, \theta)$

---

## 6. Categorical Distribution

Generalization of Bernoulli to **$K$ categories**.

$$\text{Cat}(x \mid \vec{\theta}) = \prod_{k=1}^{K} \theta_k^{[x=k]}$$

where $[x=k]$ is 1 if $x = k$, else 0.

**Constraint:** $\sum_{k=1}^K \theta_k = 1$

**One-hot encoding:** Represent category $k$ as vector $\vec{e}_k$ with 1 at position $k$:

$$P(\vec{x} = \vec{e}_k) = \theta_k$$

**ML Applications:**
- Multi-class classification: $P(y = k | \vec{x}) = \text{softmax}_k(\vec{z})$
- Language models: next-word prediction over vocabulary
- Topic models: document-topic assignment

&nbsp;

*Workout:* A classifier outputs $\vec{\theta} = (0.7, 0.2, 0.1)$ for classes (cat, dog, bird). What is $P(\text{dog})$?

**Solution:**
$P(\text{dog}) = \theta_2 = 0.2$

The predicted class is **cat** (highest probability: 0.7).

---

## 7. Multinomial Distribution

Generalization of Binomial to **$K$ categories** over $n$ trials.

$$\text{Mult}(\vec{m} \mid n, \vec{\theta}) = \frac{n!}{m_1! m_2! \cdots m_K!} \prod_{k=1}^{K} \theta_k^{m_k}$$

where $m_k$ = count of category $k$, $\sum_k m_k = n$

**Statistics:**
- $E[m_k] = n\theta_k$
- $\text{Var}(m_k) = n\theta_k(1 - \theta_k)$

**ML Application: Bag of Words**

A document with $n$ words drawn from vocabulary of $K$ words:
$$P(\text{document}) = \text{Mult}(\vec{m} \mid n, \vec{\theta})$$

where $m_k$ = count of word $k$, $\theta_k$ = probability of word $k$.

&nbsp;

*Workout:* In 100 emails, a word appears with $\theta = 0.05$. What is the expected count and variance?

**Solution:**
- $E[m] = n\theta = 100 \times 0.05 = 5$
- $\text{Var}(m) = n\theta(1-\theta) = 100 \times 0.05 \times 0.95 = 4.75$

---

## 8. Distribution Family Tree

```
                    Random Variable
                   /              \
              Discrete          Continuous
             /    |    \            |
        Bernoulli Categorical    Gaussian
        (K=2)     (K classes)    (bell curve)
           |          |              |
        Binomial  Multinomial   Multivariate
        (n trials) (n trials)    Gaussian
```

### Summary Table

| Distribution | Domain | Parameters | $E[X]$ | $\text{Var}(X)$ | ML Use |
|---|---|---|---|---|---|
| Bernoulli | $\{0,1\}$ | $\theta$ | $\theta$ | $\theta(1-\theta)$ | Binary classification |
| Binomial | $\{0,\ldots,n\}$ | $n, \theta$ | $n\theta$ | $n\theta(1-\theta)$ | Count data |
| Categorical | $\{1,\ldots,K\}$ | $\vec{\theta}$ | — | — | Multi-class output |
| Multinomial | counts | $n, \vec{\theta}$ | $n\theta_k$ | $n\theta_k(1-\theta_k)$ | Bag of words |
| Gaussian | $\mathbb{R}$ | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | Regression, latent |
| Multivariate Gaussian | $\mathbb{R}^d$ | $\vec{\mu}, \Sigma$ | $\vec{\mu}$ | $\Sigma$ | Generative models |

---

## 9. From Distributions to ML Algorithms

| ML Algorithm | Distribution Used | How |
|---|---|---|
| Logistic Regression | Bernoulli | $P(y=1|\vec{x})$ modeled as Bernoulli |
| Softmax Classifier | Categorical | $P(y=k|\vec{x})$ via softmax |
| Naive Bayes | Multinomial | Word counts in documents |
| Gaussian Mixture | Multivariate Gaussian | Cluster data with $K$ Gaussians |
| VAE | Gaussian | Latent space $\vec{z} \sim \mathcal{N}(\vec{0}, I)$ |
| Linear Regression | Gaussian | Noise assumed Gaussian |

---

## Key Takeaways

1. **Gaussian** is the workhorse: noise modeling, latent spaces, maximum entropy
2. **Bernoulli → Categorical**: binary to multi-class classification
3. **Binomial → Multinomial**: counting across categories (bag of words)
4. **Covariance matrix** $\Sigma$ connects probability to PCA/SVD from Chapter 4
5. These distributions are the building blocks for Chapter 6 (Bayesian methods)

---

## PyTorch Connection

```python
import torch
import torch.distributions as D

# Gaussian
normal = D.Normal(loc=0.0, scale=1.0)
samples = normal.sample((1000,))
log_prob = normal.log_prob(torch.tensor(0.5))

# Multivariate Gaussian
mu = torch.tensor([1.0, 2.0])
cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
mvn = D.MultivariateNormal(mu, cov)
samples_2d = mvn.sample((500,))

# Bernoulli
bern = D.Bernoulli(probs=0.7)
flips = bern.sample((100,))
print(f"Heads: {flips.sum()}")  # ≈ 70

# Categorical (e.g., softmax output)
logits = torch.tensor([2.0, 1.0, 0.5])
cat = D.Categorical(logits=logits)
classes = cat.sample((100,))

# Multinomial
probs = torch.tensor([0.5, 0.3, 0.2])
multi = D.Multinomial(total_count=100, probs=probs)
counts = multi.sample()
print(f"Counts: {counts}")  # ≈ [50, 30, 20]
```
