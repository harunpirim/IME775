# IME 775: Quiz 5 
---

### Question 1 (2 points)

A 3-class classifier outputs probability vector $\vec{y} = [0.2, 0.7, 0.1]$ for a sample whose ground truth is class 1.

**(a)** (1 pt) Compute the cross-entropy loss.

**(b)** (1 pt) If the prediction improves to $\vec{y}' = [0.05, 0.90, 0.05]$, compute the new cross-entropy loss and state by how much it decreased.

---

### Question 2 (2 points)

Compute the softmax of the score vector $\vec{s} = [1, 3, 0]$.

*Use $e^0 = 1.000$, $e^1 = 2.718$, $e^3 = 20.086$.*

---

### Question 3 (2 points)

Consider the loss function $L(w) = (w - 4)^2$ with learning rate $\eta = 0.2$ and initial weight $w_0 = 0$.

**(a)** (1 pt) Compute $w_1$ and $w_2$ using vanilla gradient descent.

**(b)** (1 pt) What value does $w_t$ converge to as $t \to \infty$?

---

### Question 4 (2 points)

A network has weights $\vec{w} = [2, -3, 1]$. The unregularized loss is $L = 1.5$ and $\lambda = 0.01$.

**(a)** (1 pt) Compute the L2 regularization penalty $R = \|\vec{w}\|^2$ and the total regularized loss.

**(b)** (1 pt) Explain in one sentence why L1 regularization tends to produce sparser weight vectors than L2 regularization.

---

### Question 5 (2 points)

**(a)** (1 pt) Name two advantages of the Adam optimizer over vanilla SGD.

**(b)** (1 pt) A network achieves 99% training accuracy but only 55% test accuracy. Name this phenomenon and describe one technique to address it.
