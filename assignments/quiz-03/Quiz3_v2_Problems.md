# IME 775 — Quiz 3 

**Name:** ______________________  

 **Q1 (2 pts)** A sensor system records readings (Low, Medium, High) across two environments (Indoor, Outdoor). The joint probability table is:

|            | Indoor | Outdoor |
|------------|--------|---------|
| **Low**    | 0.30   | 0.05    |
| **Medium** | 0.15   | 0.20    |
| **High**   | 0.05   | 0.25    |

**(a)** (1 pt) Compute the marginal distributions $P(\text{Reading})$ and $P(\text{Environment})$

**(b)** (1 pt) Compute $P(\text{High} \mid \text{Outdoor})$.


**Q2 (2 pts)** A disease screening test has the following characteristics:
- Prevalence: $P(\text{disease}) = 0.02$
- Sensitivity: $P(+ \mid \text{disease}) = 0.95$
- False positive rate: $P(+ \mid \text{no disease}) = 0.08$

**(a)** (1 pt) Compute $P(+)$ using the law of total probability.

**(b)** (1 pt) Compute $P(\text{disease} \mid +)$ using Bayes' theorem.


**Q3 (2 pts)** A random variable $X$ has the distribution $P(A) = 0.5$, $P(B) = 0.3$, $P(C) = 0.2$.

**(a)** (1 pt) Compute the Shannon entropy $H(X)$ in bits (use $\log_2$).

**(b)** (1 pt) What would $H(X)$ be if $X$ were uniformly distributed over $\{A, B, C\}$? In one sentence, explain why the uniform entropy is higher.


**Q4 (2 pts)** A 3-class classifier produces the softmax output $\hat{y} = (0.1,\ 0.8,\ 0.1)$ for an input whose true label is class 2 (i.e., the second class).

**(a)** (1 pt) Write the one-hot label vector $y$ and compute the cross-entropy loss $L = -\sum_k y_k \ln \hat{y}_k$.

**(b)** (1 pt) If the model instead output $\hat{y} = (0.2,\ 0.5,\ 0.3)$, recompute the loss. Is it higher or lower? Why?


**Q5 (2 pts)** A 2D Gaussian has mean $\boldsymbol{\mu} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ and covariance $\Sigma = \begin{pmatrix} 5 & 1 \\ 1 & 2 \end{pmatrix}$.

**(a)** (1 pt) Compute $\Sigma^{-1}$.

**(b)** (1 pt) Compute the Mahalanobis distance of the point $\mathbf{x} = (4,\ 3)^T$ from the mean.

$$d_M = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$


