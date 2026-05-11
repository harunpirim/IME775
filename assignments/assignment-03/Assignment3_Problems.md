# IME 775 — Assignment 3
## Probability Distributions & Bayesian Tools (Chapters 5–6)

**Course:** IME 775 
**Topics:** Probability Distributions, Bayes' Theorem, Entropy, Cross-Entropy, KL Divergence, MLE, MAP

---

## Part A: Probability and Distributions

**Problem 1.** A medical imaging system classifies chest X-rays into three categories with the following joint probability table over diagnosis and patient age group:

| | Young ($A_1$) | Middle ($A_2$) | Senior ($A_3$) |
|---|:---:|:---:|:---:|
| Normal ($D_1$) | 0.25 | 0.15 | 0.05 |
| Pneumonia ($D_2$) | 0.05 | 0.10 | 0.15 |
| Tumor ($D_3$) | 0.02 | 0.08 | 0.15 |

**(a)** Compute the marginal distributions $P(D)$ and $P(A)$.

**(b)** Compute $P(D_2 \mid A_3)$ — the probability of pneumonia given a senior patient.

**(c)** Are diagnosis and age group independent? Justify with a specific check.

**(d)** If you observe a senior patient, which diagnosis is most likely?

---

**Problem 2.** Let $X \sim \mathcal{N}(8, 16)$.

**(a)** State $\mu$ and $\sigma$.

**(b)** Using the 68-95-99.7 rule, find $P(4 \leq X \leq 12)$ and $P(X > 16)$.

**(c)** A second variable $Y \sim \mathcal{N}(8, 16)$ is independent of $X$. Write the joint distribution of $(X, Y)$ as a multivariate Gaussian, specifying $\vec{\mu}$ and $\Sigma$.

**(d)** Describe the shape of the equal-probability contours for this joint distribution.

---

**Problem 3.** Consider a 2D Gaussian with $\vec{\mu} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$ and $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}$.

**(a)** Compute $\Sigma^{-1}$.

**(b)** Compute the Mahalanobis distance of the point $\vec{x} = (3, 5)^T$ from the mean.

**(c)** Find the eigenvalues of $\Sigma$. What are the semi-axis lengths of the 1-$\sigma$ ellipse?

**(d)** Is the ellipse axis-aligned or rotated? Explain.

---

**Problem 4.** A sentiment classifier outputs probabilities $\vec{\theta} = (0.65, 0.25, 0.10)$ for classes (positive, neutral, negative).

**(a)** Write this as a Categorical distribution.

**(b)** In a batch of 200 reviews, what is the expected count and standard deviation for each class? (Use the Multinomial distribution.)

**(c)** If the true label is "positive," write the one-hot vector $\vec{y}$ and compute the cross-entropy loss $\mathcal{L} = -\sum_k y_k \log \hat{y}_k$.

---

**Problem 5.** A coin is flipped 20 times, resulting in 13 heads and 7 tails.

**(a)** Model each flip as $X_i \sim \text{Ber}(\theta)$. What is the MLE estimate $\hat{\theta}_{MLE}$?

**(b)** Compute $E[X]$ and $\text{Var}(X)$ using the MLE estimate.

**(c)** Using the Binomial distribution with $n = 20$ and $\hat{\theta}_{MLE}$, what is $P(\text{exactly 10 heads})$? (Set up the expression; you may leave the numerical answer in factorial/exponential form.)

---

## Part B: Bayes' Theorem and Entropy

**Problem 6.** A factory has two machines. Machine A produces 60% of all items and Machine B produces 40%. Machine A has a 3% defect rate and Machine B has a 5% defect rate.

**(a)** What is the probability that a randomly selected item is defective?

**(b)** Given that an item is defective, what is the probability it came from Machine B? Use Bayes' theorem and clearly label the prior, likelihood, and posterior.

**(c)** If 10 defective items are found, and 7 came from Machine B, update your belief using the observed data. How does this compare to your answer in (b)?

---

**Problem 7.** A discrete random variable $X$ has the following distribution:

| $x$ | A | B | C | D |
|---|:---:|:---:|:---:|:---:|
| $P(X=x)$ | 0.4 | 0.3 | 0.2 | 0.1 |

**(a)** Compute the Shannon entropy $H(X)$ in bits (use $\log_2$).

**(b)** What would the entropy be if $X$ were uniformly distributed over $\{A, B, C, D\}$? Why is this value higher?

**(c)** A sensor reading is modeled as $Y \sim \mathcal{N}(0, 9)$. Compute the differential entropy $H(Y)$ in nats.

**(d)** If the variance increases to $\sigma^2 = 25$, how does the entropy change? Explain intuitively.

---

**Problem 8.** A 3-class classifier produces softmax output $\hat{\vec{y}} = (0.7, 0.2, 0.1)$ for an input whose true label is class 1.

**(a)** Write the one-hot label vector $\vec{y}$.

**(b)** Compute the cross-entropy loss $H(\vec{y}, \hat{\vec{y}}) = -\sum_k y_k \log \hat{y}_k$ (use natural log).

**(c)** If the model instead output $\hat{\vec{y}} = (0.9, 0.05, 0.05)$, recompute the cross-entropy loss. Why is it lower?

**(d)** Show that minimizing cross-entropy loss is equivalent to maximizing the log-likelihood of the correct class.

---

## Part C: KL Divergence, MLE, and MAP

**Problem 9.** Consider two discrete distributions over $\{A, B, C\}$:
- $p = (0.5, 0.3, 0.2)$ (true distribution)
- $q = (0.4, 0.4, 0.2)$ (model distribution)

**(a)** Compute $D_{KL}(p \| q)$ in nats.

**(b)** Compute $D_{KL}(q \| p)$ in nats.

**(c)** Verify that $D_{KL}(p \| q) \neq D_{KL}(q \| p)$. What does this asymmetry mean practically?

**(d)** Compute the cross-entropy $H(p, q)$ and verify the identity $D_{KL}(p \| q) = H(p, q) - H(p)$.

---

**Problem 10.** You observe 5 data points drawn from a Gaussian distribution: $\{3.2, 4.8, 5.1, 3.9, 4.5\}$.

**(a)** Compute the MLE estimates $\hat{\mu}_{MLE}$ and $\hat{\sigma}^2_{MLE}$.

**(b)** Write the log-likelihood function $\ell(\mu, \sigma^2)$ for a Gaussian model. Show why maximizing it yields the sample mean.

**(c)** Why is $\hat{\sigma}^2_{MLE}$ a biased estimator? What is the unbiased version?

**(d)** Explain the connection: "Minimizing MSE loss in regression is equivalent to MLE under a Gaussian noise assumption."

---

**Problem 11.** A coin is flipped 10 times, yielding 7 heads. You use a Beta prior $\text{Beta}(\alpha, \beta)$ on the bias $\theta$.

**(a)** With a uniform prior ($\alpha = 1, \beta = 1$), compute the MAP estimate $\hat{\theta}_{MAP}$.

**(b)** With a prior biased toward a fair coin ($\alpha = 5, \beta = 5$), compute $\hat{\theta}_{MAP}$.

**(c)** Compare both MAP estimates with the MLE estimate $\hat{\theta}_{MLE} = 0.7$. Explain the shrinkage effect of the prior.

**(d)** Explain the analogy: "Gaussian prior on neural network weights ↔ L2 regularization."

---

**Problem 12.** Two Gaussians: $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(1, 4)$.

**(a)** Compute $D_{KL}(p \| q)$ using the closed-form formula:
$$D_{KL}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**(b)** Compute $D_{KL}(q \| p)$.

**(c)** Which direction has larger KL divergence? Give an intuitive explanation.

**(d)** In a VAE, the encoder produces $q = \mathcal{N}(\mu, \sigma^2)$ and we regularize toward $p = \mathcal{N}(0, 1)$. Write the KL regularization term and explain its role.

---

## Part D: Connections and Conceptual Questions

**Problem 13.** For each ML scenario, identify which loss function / estimation method is being used and explain the probabilistic justification:

**(a)** Training a logistic regression model with binary cross-entropy loss.

**(b)** Training a neural network with MSE loss and weight decay ($\lambda \|\vec{w}\|^2$).

**(c)** A language model that predicts the next word from a vocabulary of 50,000 words.

**(d)** A VAE whose loss has a reconstruction term plus a KL divergence term.

---

**Problem 14.** True or False (with justification):

**(a)** KL divergence is a true distance metric (i.e., it satisfies symmetry and triangle inequality).

**(b)** The MAP estimate with a uniform prior is identical to the MLE estimate.

**(c)** Cross-entropy is always greater than or equal to entropy: $H(p, q) \geq H(p)$.

**(d)** If two Gaussians have the same mean but different variances, their KL divergence is zero.

**(e)** Mutual information $I(X; Y)$ can be negative.

**(f)** The MLE estimate of the variance of a Gaussian divides by $N$ rather than $N-1$, making it biased.

---

## Part E: PyTorch Implementation

**Problem 15.** Write PyTorch code to:

**(a)** Generate 1000 samples from $\mathcal{N}(5, 4)$ and compute the MLE estimates of $\mu$ and $\sigma^2$. Compare with the true values.

**(b)** Given true distribution $p = [0.25, 0.25, 0.25, 0.25]$ and model distribution $q = [0.1, 0.2, 0.3, 0.4]$, compute the cross-entropy $H(p, q)$, entropy $H(p)$, and KL divergence $D_{KL}(p \| q)$. Verify $D_{KL} = H(p,q) - H(p)$.

**(c)** Implement a function that computes the MAP estimate for a Bernoulli parameter with a Beta prior, given observed data. Test with $k=7$ heads out of $n=10$ flips, and priors $\text{Beta}(1,1)$, $\text{Beta}(5,5)$, and $\text{Beta}(10,2)$.

---

*IME 775 Assignment 3: Chapters 5–6*  
