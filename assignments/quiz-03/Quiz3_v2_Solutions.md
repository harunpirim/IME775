# IME 775 — Quiz 3 (Version 2) — Solutions
## Probability Distributions & Bayesian Tools (Chapters 5–6, Parts A–B)

---

### Question 1 (2 pts) — Joint Probability

**(a)** Marginal distributions:

$P(\text{Reading})$: sum across columns in each row.

$$P(\text{Low}) = 0.30 + 0.05 = 0.35$$
$$P(\text{Medium}) = 0.15 + 0.20 = 0.35$$
$$P(\text{High}) = 0.05 + 0.25 = 0.30$$

$P(\text{Environment})$: sum across rows in each column.

$$P(\text{Indoor}) = 0.30 + 0.15 + 0.05 = 0.50$$
$$P(\text{Outdoor}) = 0.05 + 0.20 + 0.25 = 0.50$$

**(b)**

$$P(\text{High} \mid \text{Outdoor}) = \frac{P(\text{High, Outdoor})}{P(\text{Outdoor})} = \frac{0.25}{0.50} = \boxed{0.50}$$

---

### Question 2 (2 pts) — Bayes' Theorem

**(a)** Law of total probability:

$$P(+) = P(+ \mid D)\,P(D) + P(+ \mid \overline{D})\,P(\overline{D})$$
$$= (0.95)(0.02) + (0.08)(0.98) = 0.019 + 0.0784 = \boxed{0.0974}$$

**(b)** Bayes' theorem:

$$P(D \mid +) = \frac{P(+ \mid D)\,P(D)}{P(+)} = \frac{(0.95)(0.02)}{0.0974} = \frac{0.019}{0.0974} \approx \boxed{0.195}$$

Even with a positive test, the probability of disease is only about 19.5% because the disease prevalence (prior) is very low.

---

### Question 3 (2 pts) — Shannon Entropy

**(a)**

$$H(X) = -\sum_i P(x_i)\log_2 P(x_i)$$
$$= -\bigl[0.5\log_2(0.5) + 0.3\log_2(0.3) + 0.2\log_2(0.2)\bigr]$$
$$= -\bigl[0.5(-1) + 0.3(-1.737) + 0.2(-2.322)\bigr]$$
$$= -\bigl[-0.5 - 0.521 - 0.464\bigr] = \boxed{1.485 \text{ bits}}$$

**(b)** Uniform over $\{A,B,C\}$: $P = (1/3, 1/3, 1/3)$.

$$H_{\text{uniform}} = -3 \cdot \tfrac{1}{3}\log_2\tfrac{1}{3} = \log_2 3 \approx \boxed{1.585 \text{ bits}}$$

The uniform distribution has higher entropy because entropy is maximized when all outcomes are equally likely — there is maximum uncertainty about which outcome will occur.

---

### Question 4 (2 pts) — Cross-Entropy Loss

**(a)** True label is class 2, so the one-hot vector is $y = (0, 1, 0)$.

$$L = -\sum_k y_k \ln \hat{y}_k = -\bigl[0 \cdot \ln(0.1) + 1 \cdot \ln(0.8) + 0 \cdot \ln(0.1)\bigr] = -\ln(0.8) \approx \boxed{0.223}$$

**(b)** With $\hat{y} = (0.2, 0.5, 0.3)$:

$$L = -\ln(0.5) \approx 0.693$$

The loss is **higher** (0.693 vs. 0.223) because the model assigns less probability (0.5 vs. 0.8) to the correct class. Cross-entropy loss penalizes lower confidence in the true class.

---

### Question 5 (2 pts) — 2D Gaussian and Mahalanobis Distance

$\boldsymbol{\mu} = (2, 1)^T$, $\Sigma = \begin{pmatrix} 5 & 1 \\ 1 & 2 \end{pmatrix}$.

**(a)**

For a $2 \times 2$ matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the inverse is $\frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.

$$\det(\Sigma) = 5 \cdot 2 - 1 \cdot 1 = 9$$

$$\boxed{\Sigma^{-1} = \frac{1}{9}\begin{pmatrix} 2 & -1 \\ -1 & 5 \end{pmatrix} = \begin{pmatrix} 2/9 & -1/9 \\ -1/9 & 5/9 \end{pmatrix}}$$

**(b)**

$$\mathbf{x} - \boldsymbol{\mu} = \begin{pmatrix} 4-2 \\ 3-1 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

$$(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu}) = \begin{pmatrix} 2 & 2 \end{pmatrix} \begin{pmatrix} 2/9 & -1/9 \\ -1/9 & 5/9 \end{pmatrix} \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

First, $\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$:

$$\begin{pmatrix} 2/9 & -1/9 \\ -1/9 & 5/9 \end{pmatrix}\begin{pmatrix} 2 \\ 2 \end{pmatrix} = \begin{pmatrix} (4-2)/9 \\ (-2+10)/9 \end{pmatrix} = \begin{pmatrix} 2/9 \\ 8/9 \end{pmatrix}$$

Then the dot product:

$$\begin{pmatrix} 2 & 2 \end{pmatrix}\begin{pmatrix} 2/9 \\ 8/9 \end{pmatrix} = \frac{4}{9} + \frac{16}{9} = \frac{20}{9} \approx 2.222$$

$$d_M = \sqrt{20/9} = \frac{2\sqrt{5}}{3} \approx \boxed{1.491}$$

---

*IME 775 — Mathematical Foundations of Deep Learning*
*Quiz 3 v2: Chapters 5–6, Parts A–B*
