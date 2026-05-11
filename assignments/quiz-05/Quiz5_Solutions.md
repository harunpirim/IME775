# IME 775: Quiz 5 — Solutions

**Total Points:** 10

---

### Question 1 (2 points)

**(a)** (1 pt) GT is class 1, so only the term at $j=1$ survives:

$$L = -\log(y_1) = -\log(0.7) \approx 0.357$$

**(b)** (1 pt)

$$L' = -\log(0.90) \approx 0.105$$

Decrease: $0.357 - 0.105 = 0.252$

**Grading:** 0.5 pt for correct formula, 0.5 pt for correct numerical answer in each part.

---

### Question 2 (2 points)

$$S = e^1 + e^3 + e^0 = 2.718 + 20.086 + 1.000 = 23.804$$

$$\text{softmax}(\vec{s}) = \left[\frac{2.718}{23.804},\ \frac{20.086}{23.804},\ \frac{1.000}{23.804}\right] = [0.114,\ 0.844,\ 0.042]$$

**Grading:** 1 pt for correct $S$, 1 pt for correct probability vector.

---

### Question 3 (2 points)

**(a)** (1 pt) Gradient: $\nabla L = 2(w - 4)$

$w_1 = w_0 - \eta \cdot 2(w_0 - 4) = 0 - 0.2 \cdot 2(0 - 4) = 0 + 1.6 = 1.6$

$w_2 = 1.6 - 0.2 \cdot 2(1.6 - 4) = 1.6 - 0.2 \cdot (-4.8) = 1.6 + 0.96 = 2.56$

**(b)** (1 pt) $w_t \to 4$ (the minimum of $L(w) = (w-4)^2$).

**Grading:** 0.5 pt each for $w_1$ and $w_2$; 1 pt for correct convergence value.

---

### Question 4 (2 points)

**(a)** (1 pt)

$$R = \|\vec{w}\|^2 = 4 + 9 + 1 = 14$$

$$L_{\text{total}} = 1.5 + 0.01 \times 14 = 1.5 + 0.14 = 1.64$$

**(b)** (1 pt) L1 regularization has a constant-magnitude gradient ($\pm 1$) regardless of weight magnitude, so it pushes weights to exactly zero. L2's gradient is proportional to the weight value and shrinks as the weight approaches zero, so weights get small but never reach zero.

**Grading:** 0.5 pt for $R$, 0.5 pt for $L_{\text{total}}$; 1 pt for correct explanation (must mention constant vs. proportional gradient or equivalent reasoning).

---

### Question 5 (2 points)

**(a)** (1 pt) Any two of:
- **Per-parameter adaptive learning rates** (from RMSProp component)
- **Momentum** smooths noisy gradient estimates
- **Bias correction** compensates for initialization bias in early iterations

(0.5 pt each, must name two distinct advantages)

**(b)** (1 pt) This is **overfitting** (0.5 pt). Techniques to address it (0.5 pt for naming one with brief explanation):
- L2 regularization (weight decay): penalizes large weights, encouraging simpler models
- Dropout: randomly disables neurons during training, preventing co-adaptation
- Early stopping: stop training when validation loss starts increasing
- Data augmentation: increase effective training set size
