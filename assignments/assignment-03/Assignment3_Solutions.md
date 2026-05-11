# IME 775 — Assignment 3: Solutions
## Probability Distributions & Bayesian Tools (Chapters 5–6)

---

## Part A: Probability and Distributions

### Problem 1 — Joint Probability Table

**(a)** Marginal distributions:

$P(D_1) = 0.25 + 0.15 + 0.05 = 0.45$
$P(D_2) = 0.05 + 0.10 + 0.15 = 0.30$
$P(D_3) = 0.02 + 0.08 + 0.15 = 0.25$

$P(A_1) = 0.25 + 0.05 + 0.02 = 0.32$
$P(A_2) = 0.15 + 0.10 + 0.08 = 0.33$
$P(A_3) = 0.05 + 0.15 + 0.15 = 0.35$

**(b)** Conditional probability:

$$P(D_2 \mid A_3) = \frac{P(D_2, A_3)}{P(A_3)} = \frac{0.15}{0.35} = \frac{3}{7} \approx 0.4286$$

**(c)** Independence check: $P(D_1, A_1) = 0.25$ vs $P(D_1) \cdot P(A_1) = 0.45 \times 0.32 = 0.144$.
Since $0.25 \neq 0.144$, diagnosis and age are **not independent**.

**(d)** For a senior patient ($A_3$):
- $P(D_1 \mid A_3) = 0.05 / 0.35 \approx 0.143$
- $P(D_2 \mid A_3) = 0.15 / 0.35 \approx 0.429$
- $P(D_3 \mid A_3) = 0.15 / 0.35 \approx 0.429$

Pneumonia and Tumor are equally likely (both ≈ 42.9%), and both are far more likely than Normal.

---

### Problem 2 — 1D and Multivariate Gaussian

**(a)** $\mu = 8$, $\sigma^2 = 16$, so $\sigma = 4$.

**(b)**
- $[4, 12] = [\mu - \sigma, \mu + \sigma]$, so $P(4 \leq X \leq 12) \approx 68\%$
- $16 = \mu + 2\sigma$, so $P(X > 16) \approx \frac{1 - 0.95}{2} = 2.5\%$

**(c)** Since $X$ and $Y$ are independent with the same distribution:

$$\vec{\mu} = \begin{pmatrix} 8 \\ 8 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 16 & 0 \\ 0 & 16 \end{pmatrix}$$

**(d)** Since $\Sigma = 16I$ (scalar times identity), the contours are **circles** centered at $(8, 8)$ with radius proportional to $\sqrt{16} = 4$ in each direction.

---

### Problem 3 — Multivariate Gaussian, Mahalanobis Distance

**(a)** For $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}$:

$\det(\Sigma) = 4 \times 5 - 2 \times 2 = 16$

$$\Sigma^{-1} = \frac{1}{16}\begin{pmatrix} 5 & -2 \\ -2 & 4 \end{pmatrix} = \begin{pmatrix} 5/16 & -1/8 \\ -1/8 & 1/4 \end{pmatrix}$$

**(b)** $\vec{x} - \vec{\mu} = (3-1, 5-3)^T = (2, 2)^T$

$$D_M^2 = (2, 2) \begin{pmatrix} 5/16 & -1/8 \\ -1/8 & 1/4 \end{pmatrix} \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

First: $(2, 2) \begin{pmatrix} 5/16 & -1/8 \\ -1/8 & 1/4 \end{pmatrix} = (2 \cdot 5/16 + 2 \cdot (-1/8),\; 2 \cdot (-1/8) + 2 \cdot 1/4) = (5/8 - 1/4,\; -1/4 + 1/2) = (3/8, 1/4)$

Then: $(3/8, 1/4) \cdot (2, 2) = 3/4 + 1/2 = 5/4 = 1.25$

$$D_M = \sqrt{1.25} \approx 1.118$$

**(c)** Eigenvalues of $\Sigma$: $\lambda^2 - 9\lambda + 16 = 0$

$\lambda = \frac{9 \pm \sqrt{81 - 64}}{2} = \frac{9 \pm \sqrt{17}}{2}$

$\lambda_1 = \frac{9 + \sqrt{17}}{2} \approx 6.56$, $\lambda_2 = \frac{9 - \sqrt{17}}{2} \approx 2.44$

Semi-axis lengths: $\sqrt{6.56} \approx 2.56$ and $\sqrt{2.44} \approx 1.56$.

**(d)** The ellipse is **rotated** because $\Sigma$ has non-zero off-diagonal entries ($\text{Cov}(X_1, X_2) = 2 \neq 0$). The major axis is tilted in the direction where both variables increase together (positive correlation).

---

### Problem 4 — Categorical and Multinomial

**(a)** Categorical distribution: $P(X = k) = \theta_k$ where $\vec{\theta} = (0.65, 0.25, 0.10)$ and $k \in \{\text{positive, neutral, negative}\}$.

In one-hot form: $P(\vec{x} = \vec{e}_k) = \theta_k$.

**(b)** Multinomial with $n = 200$:

| Class | $E[m_k] = n\theta_k$ | $\text{Var}(m_k) = n\theta_k(1-\theta_k)$ | $\sigma_k$ |
|---|---|---|---|
| Positive | $200 \times 0.65 = 130$ | $200 \times 0.65 \times 0.35 = 45.5$ | $\approx 6.75$ |
| Neutral | $200 \times 0.25 = 50$ | $200 \times 0.25 \times 0.75 = 37.5$ | $\approx 6.12$ |
| Negative | $200 \times 0.10 = 20$ | $200 \times 0.10 \times 0.90 = 18.0$ | $\approx 4.24$ |

**(c)** One-hot vector: $\vec{y} = (1, 0, 0)$

$$\mathcal{L} = -\sum_k y_k \log \hat{y}_k = -[1 \cdot \log(0.65) + 0 \cdot \log(0.25) + 0 \cdot \log(0.10)] = -\log(0.65) \approx 0.431 \text{ nats}$$

---

### Problem 5 — Bernoulli and Binomial

**(a)** $\hat{\theta}_{MLE} = \frac{k}{n} = \frac{13}{20} = 0.65$

**(b)** $E[X] = \hat{\theta} = 0.65$, $\text{Var}(X) = \hat{\theta}(1 - \hat{\theta}) = 0.65 \times 0.35 = 0.2275$

**(c)** Using Binomial with $n = 20$, $\theta = 0.65$:

$$P(X = 10) = \binom{20}{10} (0.65)^{10} (0.35)^{10} = 184756 \times (0.65)^{10} \times (0.35)^{10}$$

$(0.65)^{10} \approx 0.01346$, $(0.35)^{10} \approx 2.758 \times 10^{-5}$

$$P(X = 10) \approx 184756 \times 0.01346 \times 2.758 \times 10^{-5} \approx 0.0686$$

---

## Part B: Bayes' Theorem and Entropy

### Problem 6 — Bayes' Theorem

**(a)** Total defect rate (law of total probability):

$$P(D) = P(D|A) \cdot P(A) + P(D|B) \cdot P(B) = 0.03 \times 0.60 + 0.05 \times 0.40 = 0.018 + 0.020 = 0.038$$

**(b)** Bayes' theorem:

- **Prior:** $P(B) = 0.40$
- **Likelihood:** $P(D \mid B) = 0.05$
- **Evidence:** $P(D) = 0.038$

$$P(B \mid D) = \frac{P(D \mid B) \cdot P(B)}{P(D)} = \frac{0.05 \times 0.40}{0.038} = \frac{0.020}{0.038} \approx 0.526$$

**Posterior:** Given a defective item, there is a ≈52.6% chance it came from Machine B.

**(c)** The empirical proportion is $7/10 = 0.70$, which is higher than the Bayesian prediction of 0.526. With more data, we could update our prior. The discrepancy could be due to small sample size, or the actual defect rates may differ from the assumed 3% and 5%.

---

### Problem 7 — Shannon Entropy

**(a)** $H(X) = -\sum p(x) \log_2 p(x)$

$= -[0.4 \log_2 0.4 + 0.3 \log_2 0.3 + 0.2 \log_2 0.2 + 0.1 \log_2 0.1]$

$= -[0.4(-1.322) + 0.3(-1.737) + 0.2(-2.322) + 0.1(-3.322)]$

$= -[-0.529 - 0.521 - 0.464 - 0.332]$

$= 1.846 \text{ bits}$

**(b)** Uniform over 4 outcomes: $H = \log_2 4 = 2.0$ bits.

This is higher because entropy is maximized by the uniform distribution. The original distribution concentrates probability on A and B, reducing uncertainty.

**(c)** Differential entropy of $Y \sim \mathcal{N}(0, 9)$:

$$H(Y) = \frac{1}{2}\ln(2\pi e \sigma^2) = \frac{1}{2}\ln(2\pi e \cdot 9) = \frac{1}{2}\ln(54\pi e / 3)$$

More directly: $H(Y) = \frac{1}{2}\ln(2\pi e \cdot 9) = \frac{1}{2}[\ln(2\pi e) + \ln 9]$

$= \frac{1}{2}[1.8379 + 2.1972] = \frac{1}{2}(4.0351) \approx 2.518$ nats

(Using $\ln(2\pi e) \approx 2.8379$ — correction: $\ln(2\pi) \approx 1.8379$, $\ln(e) = 1$, so $\ln(2\pi e) = 2.8379$)

$H(Y) = \frac{1}{2}[2.8379 + 2.1972] = \frac{1}{2}(5.0351) \approx 2.518$ nats

**(d)** With $\sigma^2 = 25$:

$H(Y) = \frac{1}{2}\ln(2\pi e \cdot 25) = \frac{1}{2}[2.8379 + \ln 25] = \frac{1}{2}[2.8379 + 3.2189] = \frac{1}{2}(6.0568) \approx 3.028$ nats

Entropy increases because a larger variance means the distribution is more spread out — there is more uncertainty about the value.

---

### Problem 8 — Cross-Entropy Loss

**(a)** $\vec{y} = (1, 0, 0)$ (one-hot for class 1).

**(b)** $H(\vec{y}, \hat{\vec{y}}) = -[1 \cdot \ln 0.7 + 0 \cdot \ln 0.2 + 0 \cdot \ln 0.1] = -\ln 0.7 \approx 0.357$ nats

**(c)** $H(\vec{y}, \hat{\vec{y}}) = -\ln 0.9 \approx 0.105$ nats

It is lower because the model assigns higher probability (0.9 vs 0.7) to the correct class. Cross-entropy loss decreases as the model becomes more confident about the correct answer.

**(d)** With one-hot labels, $H(\vec{y}, \hat{\vec{y}}) = -\log \hat{y}_{\text{true class}}$. This is exactly $-\log P(\text{data} \mid \theta)$, i.e., the negative log-likelihood. Minimizing cross-entropy loss:

$$\min_\theta \left[-\log \hat{y}_{\text{true class}}\right] = \max_\theta \left[\log \hat{y}_{\text{true class}}\right] = \max_\theta \left[\log P(\text{data} \mid \theta)\right]$$

which is MLE.

---

## Part C: KL Divergence, MLE, and MAP

### Problem 9 — KL Divergence (Discrete)

**(a)** $D_{KL}(p \| q) = \sum_i p_i \ln \frac{p_i}{q_i}$

$= 0.5 \ln\frac{0.5}{0.4} + 0.3 \ln\frac{0.3}{0.4} + 0.2 \ln\frac{0.2}{0.2}$

$= 0.5 \ln 1.25 + 0.3 \ln 0.75 + 0.2 \ln 1.0$

$= 0.5(0.2231) + 0.3(-0.2877) + 0.2(0)$

$= 0.1116 - 0.0863 + 0 = 0.0253$ nats

**(b)** $D_{KL}(q \| p) = \sum_i q_i \ln \frac{q_i}{p_i}$

$= 0.4 \ln\frac{0.4}{0.5} + 0.4 \ln\frac{0.4}{0.3} + 0.2 \ln\frac{0.2}{0.2}$

$= 0.4 \ln 0.8 + 0.4 \ln 1.333 + 0.2 \ln 1.0$

$= 0.4(-0.2231) + 0.4(0.2877) + 0$

$= -0.0892 + 0.1151 = 0.0259$ nats

**(c)** $D_{KL}(p \| q) = 0.0253 \neq 0.0259 = D_{KL}(q \| p)$. KL divergence is **asymmetric**.

Practically: $D_{KL}(p \| q)$ measures the cost of using model $q$ when the truth is $p$ (forward KL, mode-covering). $D_{KL}(q \| p)$ measures the cost in the reverse direction (reverse KL, mode-seeking). They penalize different types of errors — forward KL penalizes placing low $q$ where $p$ is high; reverse KL penalizes placing high $q$ where $p$ is low.

**(d)** Entropy of $p$:

$H(p) = -[0.5 \ln 0.5 + 0.3 \ln 0.3 + 0.2 \ln 0.2]$

$= -[0.5(-0.6931) + 0.3(-1.2040) + 0.2(-1.6094)]$

$= -[-0.3466 - 0.3612 - 0.3219] = 1.0297$ nats

Cross-entropy:

$H(p, q) = -[0.5 \ln 0.4 + 0.3 \ln 0.4 + 0.2 \ln 0.2]$

$= -[0.5(-0.9163) + 0.3(-0.9163) + 0.2(-1.6094)]$

$= -[-0.4581 - 0.2749 - 0.3219] = 1.0549$ nats

Verification: $D_{KL}(p \| q) = H(p, q) - H(p) = 1.0549 - 1.0297 = 0.0252 \approx 0.0253$ ✓ (small rounding difference)

---

### Problem 10 — MLE for Gaussian

**(a)** Data: $\{3.2, 4.8, 5.1, 3.9, 4.5\}$, $N = 5$

$\hat{\mu}_{MLE} = \frac{3.2 + 4.8 + 5.1 + 3.9 + 4.5}{5} = \frac{21.5}{5} = 4.30$

$\hat{\sigma}^2_{MLE} = \frac{1}{5}\sum_{i=1}^{5}(x_i - 4.3)^2 = \frac{(3.2-4.3)^2 + (4.8-4.3)^2 + (5.1-4.3)^2 + (3.9-4.3)^2 + (4.5-4.3)^2}{5}$

$= \frac{1.21 + 0.25 + 0.64 + 0.16 + 0.04}{5} = \frac{2.30}{5} = 0.46$

**(b)** Log-likelihood for $N$ i.i.d. Gaussian samples:

$$\ell(\mu, \sigma^2) = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2$$

Taking $\frac{\partial \ell}{\partial \mu} = 0$:

$$\frac{1}{\sigma^2}\sum_{i=1}^N (x_i - \mu) = 0 \implies \sum_{i=1}^N x_i = N\mu \implies \hat{\mu} = \frac{1}{N}\sum_{i=1}^N x_i = \bar{x}$$

The sum of deviations from $\mu$ must be zero, which happens exactly at the sample mean.

**(c)** $\hat{\sigma}^2_{MLE}$ divides by $N$ instead of $N-1$. On average:

$$E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2 < \sigma^2$$

So it systematically underestimates the true variance. The unbiased version is:

$$s^2 = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2 = \frac{2.30}{4} = 0.575$$

**(d)** If we model $y_i = f(\vec{x}_i; \vec{w}) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, then:

$$p(y_i \mid \vec{x}_i, \vec{w}) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(y_i - f(\vec{x}_i; \vec{w}))^2}{2\sigma^2}\right)$$

The negative log-likelihood is:

$$-\ell(\vec{w}) = \frac{N}{2}\ln(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i(y_i - f(\vec{x}_i; \vec{w}))^2$$

Minimizing over $\vec{w}$, the constant terms drop out, leaving $\min_{\vec{w}} \sum_i (y_i - f(\vec{x}_i; \vec{w}))^2$, which is exactly minimizing MSE.

---

### Problem 11 — MAP Estimation

**(a)** MAP with $\text{Beta}(1, 1)$ (uniform prior):

$$\hat{\theta}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} = \frac{7 + 1 - 1}{10 + 1 + 1 - 2} = \frac{7}{10} = 0.70$$

**(b)** MAP with $\text{Beta}(5, 5)$:

$$\hat{\theta}_{MAP} = \frac{7 + 5 - 1}{10 + 5 + 5 - 2} = \frac{11}{18} \approx 0.611$$

**(c)** Comparison:

| Method | Estimate |
|---|---|
| MLE | 0.700 |
| MAP (uniform prior) | 0.700 |
| MAP (fair-coin prior) | 0.611 |

The uniform prior has no effect — MAP = MLE. The $\text{Beta}(5, 5)$ prior pulls ("shrinks") the estimate toward 0.5 (the prior mode), reflecting prior belief in a fair coin. The stronger the prior (larger $\alpha + \beta$), the more shrinkage toward the prior mode.

**(d)** MAP objective: $\hat{\vec{w}}_{MAP} = \arg\max_{\vec{w}} \left[\log P(\mathcal{D} \mid \vec{w}) + \log P(\vec{w})\right]$

With a Gaussian prior $P(\vec{w}) = \mathcal{N}(\vec{0}, \frac{1}{\lambda}I)$:

$$\log P(\vec{w}) = -\frac{\lambda}{2}\|\vec{w}\|^2 + \text{const}$$

So MAP becomes: $\min_{\vec{w}} \left[-\log P(\mathcal{D} \mid \vec{w}) + \frac{\lambda}{2}\|\vec{w}\|^2\right]$

This is exactly the loss function with L2 regularization (weight decay). The regularization strength $\lambda$ corresponds to the precision (inverse variance) of the Gaussian prior — a tighter prior means stronger regularization.

---

### Problem 12 — Gaussian KL Divergence

**(a)** $p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(1, 4)$: $\mu_1 = 0, \sigma_1 = 1, \mu_2 = 1, \sigma_2 = 2$

$$D_{KL}(p \| q) = \ln\frac{2}{1} + \frac{1 + (0-1)^2}{2 \cdot 4} - \frac{1}{2} = \ln 2 + \frac{2}{8} - 0.5 = 0.6931 + 0.25 - 0.5 = 0.4431 \text{ nats}$$

**(b)** $D_{KL}(q \| p)$: now $\mu_1 = 1, \sigma_1 = 2, \mu_2 = 0, \sigma_2 = 1$

$$D_{KL}(q \| p) = \ln\frac{1}{2} + \frac{4 + (1-0)^2}{2 \cdot 1} - \frac{1}{2} = -0.6931 + \frac{5}{2} - 0.5 = -0.6931 + 2.5 - 0.5 = 1.3069 \text{ nats}$$

**(c)** $D_{KL}(q \| p) = 1.307 \gg D_{KL}(p \| q) = 0.443$.

The reverse direction is much larger because $q$ (wide, $\sigma = 2$) assigns substantial probability to regions where $p$ (narrow, $\sigma = 1$) has very little probability. Using narrow $p$ to evaluate wide $q$ yields large surprises. In the forward direction, $p$ is narrow so it mostly samples from the center of $q$, where $q$ is reasonably large.

**(d)** VAE KL regularization: encoder outputs $q = \mathcal{N}(\mu, \sigma^2)$, regularize toward $p = \mathcal{N}(0, 1)$:

$$D_{KL}(q \| p) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

This term encourages the latent space to stay close to the standard normal prior. It prevents the encoder from collapsing to point estimates ($\sigma \to 0$) or drifting to extreme means, ensuring a smooth, well-structured latent space that can be sampled from during generation.

---

## Part D: Connections and Conceptual Questions

### Problem 13 — ML Scenarios

**(a)** **MLE with Bernoulli likelihood.** Binary cross-entropy $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ is the negative log-likelihood of a Bernoulli distribution with parameter $\hat{y} = \sigma(\vec{w}^T\vec{x} + b)$. Minimizing it is MLE.

**(b)** **MAP with Gaussian likelihood + Gaussian prior.** MSE = negative log-likelihood under Gaussian noise. Weight decay $\lambda\|\vec{w}\|^2$ = negative log of Gaussian prior $P(\vec{w}) \propto \exp(-\frac{\lambda}{2}\|\vec{w}\|^2)$. Together, this is MAP estimation.

**(c)** **MLE with Categorical likelihood.** The model outputs a Categorical distribution over 50,000 words via softmax. Cross-entropy loss = negative log-likelihood of the correct word. This is MLE for the Categorical distribution.

**(d)** **Variational inference (ELBO maximization).** The reconstruction term is the expected log-likelihood (MLE-like). The KL term regularizes the approximate posterior toward the prior $\mathcal{N}(0, I)$. Together they form the Evidence Lower Bound (ELBO), which is a lower bound on the log-evidence $\log P(\mathcal{D})$.

---

### Problem 14 — True or False

**(a)** **False.** KL divergence is asymmetric ($D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general) and does not satisfy the triangle inequality. It is not a metric.

**(b)** **True.** With a uniform prior, $\log P(\theta)$ is constant, so MAP reduces to $\arg\max_\theta \log P(\mathcal{D} \mid \theta)$, which is MLE.

**(c)** **True.** $H(p, q) = H(p) + D_{KL}(p \| q)$. Since $D_{KL}(p \| q) \geq 0$, we have $H(p, q) \geq H(p)$, with equality iff $p = q$.

**(d)** **False.** $D_{KL}(\mathcal{N}(\mu, \sigma_1^2) \| \mathcal{N}(\mu, \sigma_2^2)) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} - \frac{1}{2}$. This is zero only when $\sigma_1 = \sigma_2$. Different variances yield nonzero KL divergence.

**(e)** **False.** Mutual information is always non-negative: $I(X; Y) = D_{KL}(p(X,Y) \| p(X)p(Y)) \geq 0$, since KL divergence is non-negative.

**(f)** **True.** $\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum(x_i - \bar{x})^2$ has $E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2 < \sigma^2$, making it biased. The unbiased version divides by $N-1$.

---

## Part E: PyTorch Implementation

### Problem 15

**(a)**

```python
import torch

torch.manual_seed(42)
samples = torch.distributions.Normal(5.0, 2.0).sample((1000,))  # N(5, 4)

mu_hat = samples.mean()
sigma2_hat = samples.var(correction=0)  # MLE: divide by N

print(f"True μ = 5.0,  MLE μ̂ = {mu_hat:.4f}")
print(f"True σ² = 4.0, MLE σ̂² = {sigma2_hat:.4f}")
```

**(b)**

```python
import torch

p = torch.tensor([0.25, 0.25, 0.25, 0.25])
q = torch.tensor([0.1, 0.2, 0.3, 0.4])

H_p = -(p * torch.log(p)).sum()
H_pq = -(p * torch.log(q)).sum()
D_kl = H_pq - H_p

print(f"H(p)       = {H_p:.4f} nats")
print(f"H(p, q)    = {H_pq:.4f} nats")
print(f"D_KL(p||q) = {D_kl:.4f} nats")
print(f"Verify: H(p,q) - H(p) = {H_pq - H_p:.4f}")
```

**(c)**

```python
import torch

def map_estimate_bernoulli(k, n, alpha, beta):
    return (k + alpha - 1) / (n + alpha + beta - 2)

k, n = 7, 10
priors = [(1, 1, "Uniform"), (5, 5, "Fair-coin"), (10, 2, "Heads-biased")]

mle = k / n
print(f"MLE estimate: {mle:.4f}")
for alpha, beta, name in priors:
    theta_map = map_estimate_bernoulli(k, n, alpha, beta)
    print(f"MAP ({name}, α={alpha}, β={beta}): {theta_map:.4f}")
```

Expected output:
```
MLE estimate: 0.7000
MAP (Uniform, α=1, β=1): 0.7000
MAP (Fair-coin, α=5, β=5): 0.6111
MAP (Heads-biased, α=10, β=2): 0.8000
```

---

*IME 775 Assignment 3 Solutions*
