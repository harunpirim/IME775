# IME 775 — Quiz 5 Solutions

## Training Neural Networks (Chapters 7–8)

**Total:** 10 points

---

### Question 1 (2 points)

**(a)** (1 pt)

$$\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{2} = \boxed{0.5}$$

$$\sigma'(0) = \sigma(0)(1 - \sigma(0)) = 0.5 \times 0.5 = \boxed{0.25}$$

**Grading:** 0.5 pts for each correct value.

**(b)** (1 pt)

$$\sigma(1;\ 3,\ 0) = \frac{1}{1 + e^{-3}} = \frac{1}{1 + 0.0498} = \frac{1}{1.0498} \approx \boxed{0.9526}$$

**Grading:** 1 pt for correct answer (accept values in range 0.950–0.953). 0.5 pts if correct setup but arithmetic error.

---

### Question 2 (2 points)

**(a)** (1 pt)

$$\vec{z}^{(0)} = \begin{pmatrix} 1 & -1 \\ 0.5 & 0.5 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 - 1 \\ 0.5 + 0.5 \end{pmatrix} = \boxed{\begin{pmatrix} 0 \\ 1 \end{pmatrix}}$$

$$\vec{a}^{(0)} = \begin{pmatrix} \sigma(0) \\ \sigma(1) \end{pmatrix} = \boxed{\begin{pmatrix} 0.5 \\ 0.7311 \end{pmatrix}}$$

**Grading:** 0.5 pts for correct $\vec{z}^{(0)}$, 0.5 pts for correct $\vec{a}^{(0)}$.

**(b)** (1 pt)

$$z^{(1)} = 1(0.5) + (-1)(0.7311) + 0 = 0.5 - 0.7311 = \boxed{-0.2311}$$

$$y = \sigma(-0.2311) \approx \boxed{0.4425}$$

**Grading:** 0.5 pts for correct $z^{(1)}$, 0.5 pts for correct $y$. Accept $y$ in range 0.440–0.445.

---

### Question 3 (2 points)

**(a)** (1 pt)

$$\ell = \frac{1}{2}(1.0 - 0.7)^2 = \frac{1}{2}(0.3)^2 = \frac{1}{2}(0.09) = \boxed{0.045}$$

**Grading:** 1 pt for correct answer. 0.5 pts if forgot the $\frac{1}{2}$ factor.

**(b)** (1 pt)

$$\delta = -(1.0 - 0.7) \times 0.7 \times (1 - 0.7) = -0.3 \times 0.7 \times 0.3 = -0.3 \times 0.21 = \boxed{-0.063}$$

**Grading:** 1 pt for correct value with correct sign. 0.5 pts for correct magnitude but wrong sign.

---

### Question 4 (2 points)

**(a)** (1 pt)

$$\frac{\partial \ell}{\partial w^{(1)}} = \delta^{(1)} \cdot a^{(0)} = (-0.05)(0.622) = \boxed{-0.0311}$$

**Grading:** 1 pt for correct answer.

**(b)** (1 pt)

$$\delta^{(0)} = (-0.05)(0.7)(0.235) = -0.035 \times 0.235 = \boxed{-0.008225}$$

$$\frac{\partial \ell}{\partial w^{(0)}} = (-0.008225)(1.0) = \boxed{-0.008225}$$

**Grading:** 0.5 pts for correct $\delta^{(0)}$, 0.5 pts for correct gradient. Accept rounding to $-0.00823$ or $-0.0082$.

---

### Question 5 (2 points)

**(a)** **False.** XOR is not linearly separable. A single perceptron can only implement linearly separable functions.

**(b)** **True.** $\sigma'(x) = \sigma(x)(1-\sigma(x))$. This product of $p$ and $(1-p)$ is maximized when $p = 0.5$, giving $0.5 \times 0.5 = 0.25$, at $x = 0$ where $\sigma(0) = 0.5$.

**(c)** **False.** In backpropagation, gradients are propagated from the **output layer toward the input layer** (backward), not from input to output. "Back" in backpropagation refers to this reverse direction.

**(d)** **True.** `loss.backward()` uses the autograd engine to compute gradients of the loss with respect to all parameters that have `requires_grad=True`, implementing the backpropagation algorithm.

**Grading:** 0.5 pts each. Answer alone is sufficient; justification is not required for full credit but earns partial credit if the T/F answer is wrong but the reasoning shows understanding.
