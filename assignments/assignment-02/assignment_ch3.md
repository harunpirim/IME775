# IME 775 – Assignment 2 (Chapter 3)
## Classifiers and Vector Calculus
**Student Name:** ______________________  


### Instructions
- Show your work for all calculations.
- You may use Python/NumPy for arithmetic, but you must show formulas and steps.
- Round final numeric answers to 4 decimal places unless otherwise stated.
- Submit a single PDF or Markdown file.

---

## Part A: Image Classification and Decision Boundaries

**1)** Explain the process of *rasterization* as described in Chapter 3. Given a tiny 3×3 grayscale image
```
X = [[10, 20, 30],
     [40, 50, 60],
     [70, 80, 90]]
```
write out the rasterized vector $\vec{x}$. What is its dimensionality?

**2)** In 3–5 sentences, explain what a *decision boundary* is in the context of a binary classifier. Include the following ideas:
- How input data points are viewed geometrically.
- What determines which class a new input is assigned to.
- The difference between a linear (hyperplane) and a nonlinear (curved) decision boundary.

**3)** The two stages of building a machine learning model are *model architecture selection* and *model training*. Define each in 2–3 sentences and explain how they relate to each other.

**4)** Consider a binary classifier with model function $q(\vec{x}; \vec{w}, b) = \vec{w}^T \vec{x} + b$ where $\vec{w} = [2, -3]^T$ and $b = 1$. For a test point $\vec{x}_0 = [1, 2]^T$:

4a) Compute $q(\vec{x}_0; \vec{w}, b)$.  
4b) If the decision rule is: class A when $q > 0$, class B when $q \leq 0$, which class does $\vec{x}_0$ belong to?  
4c) Compute $q$ for $\vec{x}_1 = [3, 1]^T$. Which class?

---

## Part B: Loss Functions and Partial Derivatives

**5)** A machine learning model makes predictions $\hat{y}_i = q(\vec{x}_i; \vec{w}, b)$ on three training examples. The known target outputs (ground truth) are $\bar{y}_i$. The squared-error loss on one example is $e_i^2 = (\hat{y}_i - \bar{y}_i)^2$, and the total loss is

$$L(\vec{w}, b) = \sum_{i=1}^{3} e_i^2$$

Given:

| $i$ | $\hat{y}_i$ | $\bar{y}_i$ |
|-----|-------------|-------------|
| 1   | 0.5         | 1.0         |
| 2   | −0.3        | −0.2        |
| 3   | 0.8         | 0.6         |

Compute the total loss $L$.

**6)** Consider the loss function $L(w_0, w_1) = 2w_0^2 + 3w_1^2$.

6a) Compute the partial derivatives $\frac{\partial L}{\partial w_0}$ and $\frac{\partial L}{\partial w_1}$.  
6b) Write the gradient vector $\nabla L(\vec{w})$.  
6c) Evaluate the gradient at the point $(w_0, w_1) = (1, -2)$.

**7)** Using the same loss function $L(w_0, w_1) = 2w_0^2 + 3w_1^2$:

7a) Compute the magnitude (L2 norm) of the gradient at $(1, -2)$.  
7b) In which direction does the gradient point — toward increasing or decreasing $L$?  
7c) In which direction should we move in parameter space to *decrease* $L$ most rapidly?

**8)** Consider $L(w_0, w_1) = w_0^2 + 4w_1^2$. At the point $(w_0, w_1) = (3, 1)$, a displacement $\Delta \vec{w} = [-0.1,\ 0.2]^T$ is applied.

8a) Compute the gradient $\nabla L$ at $(3, 1)$.  
8b) Use the first-order approximation $\Delta L \approx (\nabla L)^T \Delta \vec{w}$ to estimate the change in loss.  
8c) Compute the exact change in loss: $L(2.9, 1.2) - L(3, 1)$.  
8d) Comment on the quality of the approximation.

---

## Part C: Gradient Descent

**9)** Write out Algorithm 3.1 (gradient descent) from Chapter 3 in your own words. Your description must include:
- What the input to the algorithm is.
- How weights are updated at each iteration.
- The role of the learning rate $\eta$.
- When the algorithm terminates.

**10)** Perform **three iterations** of gradient descent on

$$L(w_0, w_1) = 2w_0^2 + 3w_1^2$$

starting from $\vec{w}^{(0)} = [1,\ -2]^T$ with learning rate $\eta = 0.1$.

For each iteration $k = 0, 1, 2$, report:
- The gradient $\nabla L(\vec{w}^{(k)})$
- The updated weights $\vec{w}^{(k+1)} = \vec{w}^{(k)} - \eta \nabla L(\vec{w}^{(k)})$
- The loss $L(\vec{w}^{(k+1)})$

Present your results in a table.

**11)** Based on your results in Problem 10:

11a) Is the loss decreasing at every step?  
11b) What would happen if you used a much larger learning rate, e.g., $\eta = 1.5$? Explain without computing.

**12)** Explain in 3–5 sentences why the gradient is zero at the minimum of a loss function and how this fact is used as a stopping criterion in gradient descent.

---

## Part D: First-Order Taylor Approximation

**13)** The one-dimensional Taylor series around a point $w$ is:

$$L(w + \Delta w) = L(w) + \Delta w \cdot \frac{dL}{dw} + \frac{(\Delta w)^2}{2!}\frac{d^2L}{dw^2} + \cdots$$

13a) Write the *first-order approximation* by keeping only the first-derivative term.  
13b) For $f(x) = x^3$, compute the first-order approximation of $f(2.1)$ expanded around $x = 2$.  
13c) Compute the exact value $f(2.1) = (2.1)^3$ and the approximation error.

**14)** For a multivariable function, the first-order Taylor approximation is:

$$\Delta L \approx (\nabla L)^T \Delta \vec{w}$$

Consider $L(w_0, w_1) = w_0 w_1 + w_0^2$ at the point $(w_0, w_1) = (1, 3)$.

14a) Compute the gradient $\nabla L$ at $(1, 3)$.  
14b) A displacement $\Delta \vec{w} = [0.05,\ -0.1]^T$ is applied. Use the first-order approximation to estimate $\Delta L$.  
14c) Compute the exact $\Delta L = L(1.05, 2.9) - L(1, 3)$ and compare.

---

## Part E: Level Contours and the Gradient

**15)** Consider $L(w_0, w_1) = w_0^2 + w_1^2$.

15a) Describe the shape of the level contours $L = c$ for various constants $c > 0$.  
15b) Compute the gradient $\nabla L$ at the point $(3, 4)$.  
15c) Show that the gradient is perpendicular to the level contour through $(3, 4)$.  
*Hint:* The tangent direction to the circle $w_0^2 + w_1^2 = 25$ at $(3,4)$ is $[-4, 3]^T$. Verify using the dot product.  
15d) Explain why moving along the gradient direction corresponds to the steepest increase in $L$.

**16)** Now consider $L(w_0, w_1) = w_0^2 + 4w_1^2$ (an elliptic paraboloid).

16a) Describe the shape of the level contours $L = c$.  
16b) Compute the gradient $\nabla L$ at $(2, 1)$.  
16c) Verify numerically that the gradient is perpendicular to the level contour at $(2, 1)$.  
*Hint:* The level contour through $(2, 1)$ is the ellipse $w_0^2 + 4w_1^2 = 8$. Implicitly differentiate to find the tangent direction.  
16d) If you perform one gradient descent step with $\eta = 0.1$, what are the new weights?

---

