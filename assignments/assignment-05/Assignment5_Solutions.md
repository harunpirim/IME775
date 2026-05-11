# IME 775: Problem Set — Loss, Optimization, and Regularization — Solutions

## Chapter 9 Practice Problems — Solutions

---

## Part I: Loss Functions

### Problem 1: Regression Loss

**(a)** $L = \|\vec{y} - \bar{y}\|^2 = (2.1-2.0)^2 + (-0.5-0)^2 + (1.8-1.5)^2 = 0.01 + 0.25 + 0.09 = 0.35$

**(b)** $\frac{\partial L}{\partial \vec{y}} = 2(\vec{y} - \bar{y}) = 2[0.1, -0.5, 0.3] = [0.2, -1.0, 0.6]$

**(c)** Using chain rule: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y_0}\cdot\frac{\partial y_0}{\partial w} = 2(y_0 - \bar{y}_0)\cdot x_0 = 2(0.1)(3.0) = 0.6$

---

### Problem 2: Cross-Entropy Loss

**(a)** $\bar{y} = [0, 0, 1, 0, 0]$ (one-hot with 1 at index 2)

**(b)** $L = -\sum_j \bar{y}_j \log(y_j) = -1\cdot\log(0.60) = -\log(0.60) \approx 0.5108$

**(c)** $L' = -\log(0.92) \approx 0.0834$. Decrease: $0.5108 - 0.0834 = 0.4274$

**(d)** Minimum CE loss is $0$ when $y_{j^*} = 1$, i.e., $\vec{y} = [0, 0, 1, 0, 0]$ (prediction matches GT exactly with probability 1).

---

### Problem 3: Binary Cross-Entropy

**(a)** With $\bar{y} = 1$: $L = -\log(y)$

| $y$ | $L = -\log(y)$ |
|---|---|
| 0.1 | 2.3026 |
| 0.3 | 1.2040 |
| 0.5 | 0.6931 |
| 0.7 | 0.3567 |
| 0.9 | 0.1054 |
| 0.99 | 0.0101 |

**(b)** The curve is $L = -\log(y)$ for $y \in (0,1)$: it goes from $\infty$ at $y=0$ to $0$ at $y=1$, decreasing and convex.

**(c)** $L = -\bar{y}\log(y) - (1-\bar{y})\log(1-y)$

$$\frac{\partial L}{\partial y} = -\frac{\bar{y}}{y} + \frac{1-\bar{y}}{1-y} = 0$$

$$\frac{\bar{y}}{y} = \frac{1-\bar{y}}{1-y} \implies \bar{y}(1-y) = y(1-\bar{y}) \implies \bar{y} - \bar{y}y = y - y\bar{y} \implies y = \bar{y}$$

---

## Part II: Softmax

### Problem 4: Softmax Computation

**(a)** $e^3 = 20.086$, $e^1 = 2.718$, $e^{-1} = 0.368$, $e^0 = 1.000$

$S = 20.086 + 2.718 + 0.368 + 1.000 = 24.171$

$\text{softmax}(\vec{s}) = [0.831, 0.112, 0.015, 0.041]$

**(b)** $L = -\log(0.831) \approx 0.185$

**(c)** $e^6 = 403.4$, $e^2 = 7.389$, $e^{-2} = 0.135$, $e^0 = 1.000$

$S = 411.9$

$\text{softmax}(\vec{s}') = [0.979, 0.018, 0.0003, 0.002]$

The network is **more confident** (probability of class 0 increased from 0.831 to 0.979). The CE loss decreased to $-\log(0.979) \approx 0.021$.

Doubling the scores is equivalent to temperature $\tau = 0.5$, which makes softmax sharper.

**(d)** With $\tau = 0.1$: softmax becomes nearly one-hot (very confident). With $\tau = 10$: softmax becomes nearly uniform $[0.25, 0.25, 0.25, 0.25]$ (maximum uncertainty).

---

### Problem 5: Softmax Properties

**(a)** $\sum_j \frac{e^{s_j}}{S} = \frac{1}{S}\sum_j e^{s_j} = \frac{S}{S} = 1$ ∎

**(b)** $\text{softmax}(\vec{s}+c)_j = \frac{e^{s_j+c}}{\sum_k e^{s_k+c}} = \frac{e^c\,e^{s_j}}{e^c\sum_k e^{s_k}} = \frac{e^{s_j}}{\sum_k e^{s_k}} = \text{softmax}(\vec{s})_j$ ∎

**Numerical stability:** We subtract $\max(\vec{s})$ from all scores before exponentiation. This prevents overflow from $e^{s_j}$ when $s_j$ is large, without changing the result.

**(c)** Let $p_i = \text{softmax}(\vec{s})_i$. Then:

$$\frac{\partial p_i}{\partial s_j} = \begin{cases} p_i(1-p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

In compact form: $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$

---

## Part III: Focal Loss and Hinge Loss

### Problem 6: Focal Loss

**(a)** $L = -(1-y_t)^\gamma \log(y_t)$
- $\gamma=0$: $L = -\log(y_t)$ (standard CE)
- $\gamma=1$: $L = -(1-y_t)\log(y_t)$
- $\gamma=2$: $L = -(1-y_t)^2\log(y_t)$

**(b)**

| $y_t$ | $\gamma=0$ | $\gamma=1$ | $\gamma=2$ |
|---|---|---|---|
| 0.1 | 2.303 | 2.072 | 1.865 |
| 0.5 | 0.693 | 0.347 | 0.173 |
| 0.9 | 0.105 | 0.011 | 0.001 |

**(c)** $\frac{0.001}{0.105} \approx 0.010 = 1\%$. Only 1% of the standard CE loss remains for the easy example, effectively ignoring it during training.

**(d)** Most examples are class 0 (easy to classify correctly). Without focal loss, the gradient is dominated by these easy examples, and the network makes little progress on the rare class 1 examples. Focal loss down-weights the easy class 0 examples and amplifies the contribution of the harder class 1 examples, so training focuses on improving the minority class.

---

### Problem 7: Multi-class Hinge Loss

**(a)** GT class $c = 1$, $y_c = 8$, $m = 1$:

- $j=0$: $\max(0, 5-8+1) = \max(0, -2) = 0$
- $j=2$: $\max(0, 3-8+1) = \max(0, -4) = 0$
- $j=3$: $\max(0, 2-8+1) = \max(0, -5) = 0$

$L = 0 + 0 + 0 = 0$

**(b)** No class contributes because every incorrect class score is at least $m=1$ below the correct class score. The condition $y_j < y_c - m$ is satisfied for all $j \neq c$.

**(c)** The correct class score must satisfy $y_c \geq y_j + m$ for all $j \neq c$. The maximum incorrect score is $y_0 = 5$. So $y_c \geq 5 + 1 = 6$.

**(d)** Both have hinge loss = 0. Once the correct class wins by the margin, increasing its score further doesn't change the hinge loss. CE loss, by contrast, would decrease from $-\log(0.831)$ to $-\log(\approx 1.0)$ — CE always rewards increased confidence.

---

## Part IV: Optimization

### Problem 8: Vanilla SGD

**(a)** $\frac{\partial L}{\partial w} = 2(w-3)$

**(b)** Update: $w_{t+1} = w_t - 0.1 \cdot 2(w_t - 3) = w_t - 0.2(w_t - 3) = 0.8w_t + 0.6$

| $t$ | $w_t$ | $L(w_t)$ |
|---|---|---|
| 0 | 0.000 | 9.000 |
| 1 | 0.600 | 5.760 |
| 2 | 1.080 | 3.686 |
| 3 | 1.464 | 2.359 |
| 4 | 1.771 | 1.510 |
| 5 | 2.017 | 0.966 |

**(c)** $w_t \to 3$ (the minimum of $L$). The recurrence $w_{t+1} = 0.8w_t + 0.6$ has fixed point $w^* = 0.6/(1-0.8) = 3$.

**(d)** With $\eta = 1.0$: $w_{t+1} = w_t - 2(w_t-3) = -w_t + 6$. This gives $w_0=0, w_1=6, w_2=0, w_3=6, \ldots$ — oscillation, no convergence.

With $\eta = 1.1$: $w_{t+1} = w_t - 2.2(w_t-3) = -1.2w_t + 6.6$. Values: $0, 6.6, -1.32, 8.184, \ldots$ — divergence.

---

### Problem 9: Momentum

**(a)** $\Delta w_t = 0.9\,\Delta w_{t-1} + 0.1\cdot 2(w_t-3)$, then $w_{t+1} = w_t - \Delta w_t$

| $t$ | $w_t$ | $\nabla L$ | $\Delta w_t$ | $L(w_t)$ |
|---|---|---|---|---|
| 0 | 0.000 | -6.0 | -0.6 | 9.000 |
| 1 | 0.600 | -4.8 | -1.02 | 5.760 |
| 2 | 1.620 | -2.76 | -1.194 | 1.904 |
| 3 | 2.814 | -0.372 | -1.112 | 0.035 |
| 4 | 3.926 | 1.852 | -0.815 | 0.857 |
| 5 | 4.741 | 3.482 | -0.385 | 3.031 |

Note: Momentum overshoots the minimum ($w=3$) at $t=3$ and oscillates. However, it reaches the vicinity of the minimum much faster than vanilla SGD.

**(b)** Vanilla SGD $w_5 = 2.017$; Momentum $w_5 = 4.741$. Momentum reaches and passes the minimum much faster, but overshoots. With more iterations, both converge to $w=3$, but momentum oscillates more near the minimum.

**(c)** $\beta = 0$: reduces to vanilla SGD (no memory of past). $\beta = 0.99$: very strong momentum — the ball rolls very fast and may overshoot significantly, requiring many oscillations to settle.

---

### Problem 10: Adam Optimizer

**(a)** Bias correction:

$$\hat{v}_3 = \frac{v_3}{1-\beta_1^3} = \frac{0.15}{1-0.9^3} = \frac{0.15}{1-0.729} = \frac{0.15}{0.271} \approx 0.5535$$

$$\hat{s}_3 = \frac{s_3}{1-\beta_2^3} = \frac{0.04}{1-0.999^3} = \frac{0.04}{1-0.997} = \frac{0.04}{0.003} \approx 13.333$$

**(b)** $\Delta w_3 = 0.001\cdot\frac{0.5535}{\sqrt{13.333}+10^{-8}} = 0.001\cdot\frac{0.5535}{3.651} \approx 0.0001516$

**(c)** In early iterations, $v_t$ and $s_t$ are initialized to zero and heavily biased toward zero because only a few gradient values have been accumulated. Bias correction inflates them to their true expected magnitudes. As $t \to \infty$, $\beta_1^t \to 0$ and $\beta_2^t \to 0$, so the correction factors $(1-\beta^t) \to 1$ and $\hat{v}_t \to v_t$, $\hat{s}_t \to s_t$.

---

## Part V: Regularization and Dropout

### Problem 11: L2 Regularization

**(a)** $R = \|\vec{w}\|^2 + \|\vec{b}\|^2 = (9+4+1+0.25) + (0.01+0.04) = 14.25 + 0.05 = 14.30$

**(b)** $L_{\text{total}} = 2.5 + 0.01 \times 14.30 = 2.5 + 0.143 = 2.643$

**(c)** $\frac{\partial R}{\partial w_0} = 2w_0 = 2(3) = 6$. The gradient pushes $w_0$ toward zero proportional to its magnitude. The total gradient contribution from regularization is $\lambda \cdot 6 = 0.06$, which gets subtracted from $w_0$ at each update.

**(d)** $L_{\text{total}} = 2.5 + 1.0 \times 14.30 = 16.80$. With large $\lambda$, the regularization term dominates, and the optimizer focuses on shrinking weights rather than minimizing the actual training loss. This leads to underfitting — the network is too constrained to fit the data.

---

### Problem 12: L1 vs. L2 Regularization

**(a)** $L_1(w) = (w-0.5)^2 + 0.1|w|$

**(b)** $L_2(w) = (w-0.5)^2 + 0.1w^2$

**(c)** $\frac{dL_2}{dw} = 2(w-0.5) + 0.2w = 2.2w - 1 = 0$

$w^* = \frac{1}{2.2} \approx 0.4545$

L2 shrinks $w^*$ from $0.5$ toward zero but doesn't reach zero.

**(d)** For $w > 0$: $L_1(w) = (w-0.5)^2 + \lambda w$

$\frac{dL_1}{dw} = 2(w-0.5) + \lambda = 0 \implies w^* = 0.5 - \lambda/2$

For $w^* = 0$: $0.5 - \lambda/2 = 0 \implies \lambda = 1.0$

For $\lambda \geq 1.0$, the optimal $w^*$ is exactly 0. With L2, the solution $w^* = 0.5/(1+\lambda)$ approaches zero but never reaches it.

---

### Problem 13: Dropout

**(a)** $2^4 = 16$ distinct subnetworks

**(b)** Output = $[2.0, 0, 0.5, 0] \odot \frac{1}{p} = [4.0, 0, 1.0, 0]$ (PyTorch uses inverted dropout, dividing kept values by $p$)

**(c)** During training with dropout probability $p = 0.5$, each neuron is active only half the time. At inference, all neurons are active, so their raw outputs are effectively doubled compared to training. To compensate: either scale down by $p$ at inference (standard dropout) or scale up by $1/p$ during training (inverted dropout, used by PyTorch). Both ensure the expected output magnitude is consistent.

**(d)** During training, $E[\text{output}_j] = p \cdot \frac{a_j}{p} + (1-p)\cdot 0 = a_j$ (with inverted dropout).

For $a_0 = 2.0$: $E = 0.5 \cdot 4.0 + 0.5 \cdot 0 = 2.0$ ✓ Matches inference output $2.0$.

---

### Problem 14: Bayesian Interpretation

**(a)** Bayes: $p(\theta|T) \propto p(T|\theta)\,p(\theta)$

Take negative log: $-\log p(\theta|T) = -\log p(T|\theta) - \log p(\theta) + \text{const}$

With $p(T|\theta) \propto e^{-L(\theta)}$: $-\log p(T|\theta) = L(\theta)$

With Gaussian prior $p(\theta) \propto e^{-\lambda\|\theta\|^2}$: $-\log p(\theta) = \lambda\|\theta\|^2$

Maximizing $p(\theta|T)$ ↔ minimizing $-\log p(\theta|T) = L(\theta) + \lambda\|\theta\|^2$, which is L2-regularized loss. ∎

**(b)** L1 regularization corresponds to a Laplace prior: $p(\theta) \propto e^{-\lambda|\theta|}$. This is the **Laplace distribution** (double exponential), which has a sharp peak at zero and heavy tails.

**(c)** Large $\lambda$ means the prior is very tight around zero — we strongly believe the parameters should be small. This constrains the model to find solutions with small weights, potentially at the cost of fitting the training data less well. Choosing $\lambda$ trades off data fit vs. prior belief.

---

### Problem 15: Putting It All Together

**(a)** **Focal loss** is the best choice. The dataset is highly imbalanced (1000 vs 50 vs 50). Standard CE loss would be dominated by the majority class. Focal loss down-weights the easy majority class examples and emphasizes the harder minority class examples. (Using weighted CE with class weights inversely proportional to frequency would also be acceptable.)

**(b)** Start with **Adam** optimizer, $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$. Adam is the modern default: it adapts per-parameter learning rates and includes momentum.

**(c)** This is **overfitting**. The network (two 128-neuron hidden layers) has enough capacity to memorize all 1100 training examples. Training loss near zero means the network has memorized the training data. High test error means it hasn't learned generalizable patterns.

**(d)**

1. **L2 regularization** (weight decay $\lambda = 10^{-4}$): Penalizes large weights, encouraging simpler decision boundaries that generalize better.

2. **Dropout** ($p = 0.5$ after each hidden layer): Prevents co-adaptation of neurons and simulates an ensemble of subnetworks, each of which is simpler than the full network.

**(e)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 3),
)

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 20.0, 20.0]))
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        scores = model(X_batch)
        loss = loss_fn(scores, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
