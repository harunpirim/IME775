# IME 775 – Assignment 2 Solutions (Chapter 3)
## Classifiers and Vector Calculus

---

## Part A: Image Classification and Decision Boundaries (16 pts)

### Problem 1 (4 pts)

Rasterization converts a 2-D image (matrix) into a 1-D vector by reading elements left-to-right, top-to-bottom.

$$\vec{x} = [10,\ 20,\ 30,\ 40,\ 50,\ 60,\ 70,\ 80,\ 90]^T$$

Dimensionality = $3 \times 3 = 9$.

---

### Problem 2 (4 pts)

*Sample answer:*

In a classification problem every input can be represented as a vector, equivalently a point in a high-dimensional feature space. Points belonging to the same class tend to form clusters. A **decision boundary** is a hypersurface that separates these clusters. To classify a new input we check which side of the boundary the point falls on.

- **Linear boundary (hyperplane):** sufficient when clusters are well separated (e.g., car vs. giraffe in the textbook example).
- **Nonlinear boundary (curved surface):** needed when clusters overlap or interleave (e.g., horse vs. zebra requiring a hypersphere).

---

### Problem 3 (4 pts)

*Sample answer:*

**Model architecture selection** — Choose a parameterized function family $q(\vec{x};\vec{w},b)$ (e.g., a linear model for simple tasks, a multilayer neural network for complex tasks). At this stage the parameters $\vec{w}, b$ are still unknown; only the functional form is fixed.

**Model training** — Estimate $\vec{w}, b$ so that the model output matches the known ground truth on the training data as closely as possible, typically by iteratively minimizing a loss function via gradient descent.

The two stages are sequential: architecture selection defines the search space; training searches within that space.

---

### Problem 4 (4 pts)

Model: $q(\vec{x}) = \vec{w}^T\vec{x} + b = 2x_0 - 3x_1 + 1$.

**4a)** $q(\vec{x}_0) = 2(1) - 3(2) + 1 = 2 - 6 + 1 = -3$.

**4b)** $q = -3 \leq 0$ → **Class B**.

**4c)** $q(\vec{x}_1) = 2(3) - 3(1) + 1 = 6 - 3 + 1 = 4 > 0$ → **Class A**.

---

## Part B: Loss Functions and Partial Derivatives (20 pts)

### Problem 5 (4 pts)

| $i$ | $\hat{y}_i$ | $\bar{y}_i$ | $e_i = \hat{y}_i - \bar{y}_i$ | $e_i^2$ |
|-----|-------------|-------------|-------------------------------|---------|
| 1   | 0.5         | 1.0         | −0.5                          | 0.25    |
| 2   | −0.3        | −0.2        | −0.1                          | 0.01    |
| 3   | 0.8         | 0.6         | 0.2                           | 0.04    |

$$L = 0.25 + 0.01 + 0.04 = \boxed{0.30}$$

---

### Problem 6 (6 pts)

$L(w_0, w_1) = 2w_0^2 + 3w_1^2$

**6a)**

$$\frac{\partial L}{\partial w_0} = 4w_0, \qquad \frac{\partial L}{\partial w_1} = 6w_1$$

**6b)**

$$\nabla L = \begin{bmatrix} 4w_0 \\ 6w_1 \end{bmatrix}$$

**6c)** At $(1, -2)$:

$$\nabla L = \begin{bmatrix} 4(1) \\ 6(-2) \end{bmatrix} = \begin{bmatrix} 4 \\ -12 \end{bmatrix}$$

---

### Problem 7 (6 pts)

**7a)** $\|\nabla L\| = \sqrt{4^2 + (-12)^2} = \sqrt{16 + 144} = \sqrt{160} \approx 12.6491$.

**7b)** The gradient points in the direction of **steepest increase** of $L$.

**7c)** To decrease $L$ most rapidly, move in the **negative gradient** direction: $-\nabla L = [-4,\ 12]^T$.

---

### Problem 8 (4 pts)

$L(w_0, w_1) = w_0^2 + 4w_1^2$

**8a)** $\nabla L = [2w_0,\ 8w_1]^T$. At $(3, 1)$: $\nabla L = [6,\ 8]^T$.

**8b)** First-order approximation:

$$\Delta L \approx \nabla L^T \Delta\vec{w} = [6,\ 8] \begin{bmatrix} -0.1 \\ 0.2 \end{bmatrix} = 6(-0.1) + 8(0.2) = -0.6 + 1.6 = \boxed{1.0}$$

**8c)** Exact:

$$L(3,1) = 9 + 4 = 13, \quad L(2.9, 1.2) = 8.41 + 5.76 = 14.17$$
$$\Delta L_{\text{exact}} = 14.17 - 13 = \boxed{1.17}$$

**8d)** The approximation (1.0) is close to the exact change (1.17); the error of 0.17 arises from the neglected higher-order terms. For smaller displacements the first-order approximation would be even more accurate.

---

## Part C: Gradient Descent (24 pts)

### Problem 9 (6 pts)

*Sample answer:*

**Input:** Loss function $L(\vec{w})$, initial weights $\vec{w}^{(0)}$, learning rate $\eta$.

**Repeat:**
1. Compute the gradient $\nabla L(\vec{w}^{(k)})$.
2. Update weights: $\vec{w}^{(k+1)} = \vec{w}^{(k)} - \eta \nabla L(\vec{w}^{(k)})$.
3. Evaluate $L(\vec{w}^{(k+1)})$.

**Terminate** when $L$ is sufficiently small or $\|\nabla L\| \approx 0$.

The learning rate $\eta$ controls step size: too large risks overshooting the minimum; too small makes convergence slow.

---

### Problem 10 (10 pts)

$L(w_0, w_1) = 2w_0^2 + 3w_1^2$, $\nabla L = [4w_0,\ 6w_1]^T$, $\eta = 0.1$.

| Step $k$ | $\vec{w}^{(k)}$ | $\nabla L(\vec{w}^{(k)})$ | $\vec{w}^{(k+1)}$ | $L(\vec{w}^{(k+1)})$ |
|-----------|-----------------|---------------------------|--------------------|-----------------------|
| 0 | $[1.000,\ -2.000]$ | $[4.000,\ -12.000]$ | $[0.600,\ -0.800]$ | 2.640 |
| 1 | $[0.600,\ -0.800]$ | $[2.400,\ -4.800]$ | $[0.360,\ -0.320]$ | 0.566 |
| 2 | $[0.360,\ -0.320]$ | $[1.440,\ -1.920]$ | $[0.216,\ -0.128]$ | 0.142 |

Starting loss: $L(\vec{w}^{(0)}) = 2(1) + 3(4) = 14$.

---

### Problem 11 (4 pts)

**11a)** Yes — $14 \to 2.640 \to 0.566 \to 0.142$; the loss decreases monotonically.

**11b)** With $\eta = 1.5$, the update steps would be very large. For instance, $w_0^{(1)} = 1 - 1.5(4) = -5$, which overshoots the minimum at $w_0 = 0$ by a wide margin. The weights would oscillate wildly and the loss could *increase*, causing divergence.

---

### Problem 12 (4 pts)

At a minimum, the function surface is flat — there is no direction in which $L$ decreases further. Mathematically, $\nabla L = \vec{0}$ at a minimum because all partial derivatives vanish (the function is at a stationary point). In gradient descent the update $\vec{w}^{(k+1)} = \vec{w}^{(k)} - \eta \nabla L$ becomes $\vec{w}^{(k+1)} = \vec{w}^{(k)}$ when $\nabla L = \vec{0}$, so the weights stop changing. Therefore, $\|\nabla L\| \approx 0$ (or $L$ ceasing to decrease) serves as a practical stopping criterion.

---

## Part D: First-Order Taylor Approximation (12 pts)

### Problem 13 (6 pts)

**13a)** First-order approximation:

$$L(w + \Delta w) \approx L(w) + \Delta w \cdot \frac{dL}{dw}$$

**13b)** $f(x) = x^3$, expand around $x = 2$ with $\Delta w = 0.1$:

$$f(2) = 8, \qquad f'(x) = 3x^2, \qquad f'(2) = 12$$
$$f(2.1) \approx 8 + 0.1 \times 12 = \boxed{9.2}$$

**13c)** Exact: $f(2.1) = 2.1^3 = 9.261$.

Approximation error $= |9.261 - 9.2| = \boxed{0.061}$.

The error is small because $\Delta w = 0.1$ is reasonably small.

---

### Problem 14 (6 pts)

$L(w_0, w_1) = w_0 w_1 + w_0^2$

$\frac{\partial L}{\partial w_0} = w_1 + 2w_0$, $\frac{\partial L}{\partial w_1} = w_0$.

**14a)** At $(1, 3)$: $\nabla L = [3 + 2,\ 1]^T = [5,\ 1]^T$.

**14b)**

$$\Delta L \approx [5,\ 1] \begin{bmatrix} 0.05 \\ -0.1 \end{bmatrix} = 0.25 - 0.1 = \boxed{0.15}$$

**14c)** Exact:

$$L(1, 3) = 3 + 1 = 4$$
$$L(1.05, 2.9) = (1.05)(2.9) + (1.05)^2 = 3.045 + 1.1025 = 4.1475$$
$$\Delta L_{\text{exact}} = 4.1475 - 4 = 0.1475$$

The first-order approximation (0.15) is very close to the exact value (0.1475); the error of 0.0025 comes from neglected higher-order terms.

---

## Part E: Convexity (12 pts)

### Problem 15 (6 pts)

**(a)** $f(x) = 3x^2 + 5$: $f''(x) = 6 > 0$ for all $x$. → **Convex** ✓

**(b)** $f(x) = e^x$: $f''(x) = e^x > 0$ for all $x$. → **Convex** ✓

**(c)** $f(x) = x^4 - 2x^2$: $f''(x) = 12x^2 - 4$. At $x = 0$: $f''(0) = -4 < 0$. Since $f''$ is negative in some regions, the function is → **Neither (non-convex)**.

**(d)** $f(x) = -\ln(x)$ for $x > 0$: $f'(x) = -1/x$, $f''(x) = 1/x^2 > 0$ for all $x > 0$. → **Convex** ✓

---

### Problem 16 (3 pts)

A **convex** loss function has a single global minimum and no local minima. Gradient descent on a convex function is guaranteed to converge to the global minimum regardless of the starting point. A **non-convex** loss function can have multiple local minima and saddle points. Gradient descent may become trapped in a local minimum that is not the global minimum, yielding a sub-optimal solution.

---

### Problem 17 (3 pts)

For a convex function, every local minimum is also the global minimum (in fact, there is only one minimum). The gradient points away from the minimum everywhere, so the negative gradient always points toward it. Therefore, gradient descent — which moves in the negative-gradient direction — always makes progress toward the unique global minimum and will converge there given a sufficiently small learning rate.

---

## Part F: Level Contours and the Gradient (16 pts)

### Problem 18 (8 pts)

$L(w_0, w_1) = w_0^2 + w_1^2$

**18a)** The level contours $w_0^2 + w_1^2 = c$ are **circles** centered at the origin with radius $\sqrt{c}$.

**18b)** $\nabla L = [2w_0,\ 2w_1]^T$. At $(3, 4)$: $\nabla L = [6,\ 8]^T$.

**18c)** The level contour through $(3,4)$ is $w_0^2 + w_1^2 = 25$ (a circle of radius 5). The tangent to this circle at $(3,4)$ is in the direction $[-4,\ 3]^T$. Check perpendicularity:

$$\nabla L \cdot \vec{t} = [6,\ 8] \cdot [-4,\ 3] = -24 + 24 = 0 \quad \checkmark$$

The gradient is perpendicular to the level contour.

**18d)** Since the gradient is perpendicular to the level contour and points outward (toward higher $L$ values), moving along the gradient direction crosses level contours as quickly as possible — this is the direction of steepest increase. Moving in the *opposite* direction (negative gradient) gives steepest decrease.

---

### Problem 19 (8 pts)

$L(w_0, w_1) = w_0^2 + 4w_1^2$

**19a)** The level contours $w_0^2 + 4w_1^2 = c$ are **ellipses** centered at the origin (wider along the $w_0$ axis).

**19b)** $\nabla L = [2w_0,\ 8w_1]^T$. At $(2, 1)$: $\nabla L = [4,\ 8]^T$.

**19c)** The level contour through $(2,1)$ is $w_0^2 + 4w_1^2 = 8$. Implicit differentiation:

$$2w_0 + 8w_1 \frac{dw_1}{dw_0} = 0 \implies \frac{dw_1}{dw_0} = -\frac{w_0}{4w_1} = -\frac{2}{4} = -0.5$$

Tangent direction: $[2,\ -1]^T$ (or any scalar multiple).

$$\nabla L \cdot \vec{t} = [4,\ 8] \cdot [2,\ -1] = 8 - 8 = 0 \quad \checkmark$$

**19d)** One gradient descent step with $\eta = 0.1$:

$$\vec{w}^{(1)} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} - 0.1 \begin{bmatrix} 4 \\ 8 \end{bmatrix} = \begin{bmatrix} 1.6 \\ 0.2 \end{bmatrix}$$

New loss: $L(1.6, 0.2) = 2.56 + 0.16 = 2.72$ (down from 8).
