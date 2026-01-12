# Week 4: Second-Order Optimization - Newton's Method

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 4: Second-Order Optimization Techniques**.

---

## Overview

This week covers Newton's method and second-order optimization, which uses curvature information for faster convergence.

---

## Learning Objectives

- Understand second-order optimality conditions
- Derive and implement Newton's method
- Compare Newton's method with gradient descent
- Identify weaknesses and practical considerations

---

## 4.1 The Second-Order Optimality Condition

### The Hessian Matrix

$$\nabla^2 g(w) = H = \begin{bmatrix} 
\frac{\partial^2 g}{\partial w_1^2} & \cdots & \frac{\partial^2 g}{\partial w_1 \partial w_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 g}{\partial w_n \partial w_1} & \cdots & \frac{\partial^2 g}{\partial w_n^2}
\end{bmatrix}$$

### Conditions for Local Minimum

**Necessary conditions** (if $w^*$ is local min):
1. $\nabla g(w^*) = 0$
2. $\nabla^2 g(w^*) \succeq 0$ (positive semi-definite)

**Sufficient conditions** (guarantee local min):
1. $\nabla g(w^*) = 0$
2. $\nabla^2 g(w^*) \succ 0$ (positive definite)

### Classifying Stationary Points

| Hessian | Classification |
|---------|---------------|
| Positive definite | Local minimum |
| Negative definite | Local maximum |
| Indefinite | Saddle point |
| Singular | Need higher-order analysis |

---

## 4.2 The Geometry of Second-Order Taylor Series

### Quadratic Approximation

$$g(w + d) \approx g(w) + \nabla g(w)^T d + \frac{1}{2} d^T \nabla^2 g(w) d$$

This is a quadratic function in the step $d$.

### Minimizing the Quadratic

Take derivative with respect to $d$ and set to zero:

$$\nabla_d q(d) = \nabla g(w) + \nabla^2 g(w) d = 0$$

Solve for optimal direction:

$$d^* = -[\nabla^2 g(w)]^{-1} \nabla g(w)$$

This is the **Newton direction**.

### Geometric Interpretation

- GD: Step proportional to gradient (ignores curvature)
- Newton: Step accounts for local curvature
- Newton finds minimum of quadratic approximation

---

## 4.3 Newton's Method

### The Newton Update

$$w^{(k+1)} = w^{(k)} - [\nabla^2 g(w^{(k)})]^{-1} \nabla g(w^{(k)})$$

### Algorithm

```python
def newtons_method(grad_g, hess_g, w0, max_iter=100, tol=1e-6):
    w = np.array(w0, dtype=float)
    
    for k in range(max_iter):
        gradient = grad_g(w)
        hessian = hess_g(w)
        
        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break
        
        # Newton direction
        d = np.linalg.solve(hessian, -gradient)
        
        # Update
        w = w + d
    
    return w
```

### For Quadratic Functions

If $g(w) = \frac{1}{2}w^T A w - b^T w + c$:
- $\nabla g = Aw - b$
- $\nabla^2 g = A$

Newton's method converges in **one step**!

### Convergence Rate

Near a local minimum with positive definite Hessian:

$$\|w^{(k+1)} - w^*\| \leq C \|w^{(k)} - w^*\|^2$$

**Quadratic convergence**: The number of correct digits approximately doubles each iteration.

---

## Comparison: Newton vs Gradient Descent

| Aspect | Gradient Descent | Newton's Method |
|--------|-----------------|-----------------|
| Update | $w - \alpha \nabla g$ | $w - H^{-1} \nabla g$ |
| Convergence | Linear | Quadratic |
| Cost per step | $O(n)$ | $O(n^3)$ |
| Memory | $O(n)$ | $O(n^2)$ |
| Learning rate | Required | Not needed (full step) |
| Curvature | Ignored | Used |

### When to Use Newton

- Small to medium problems ($n < 1000$)
- When second derivatives are cheap
- Near optimum (for fast final convergence)

### When to Use GD

- Large-scale problems ($n > 10^6$)
- When Hessian is expensive/unavailable
- When approximate solution is acceptable

---

## 4.4 Two Natural Weaknesses of Newton's Method

### Weakness 1: Computational Cost

**Hessian computation**: $O(n^2)$ elements
**Solving linear system**: $O(n^3)$ operations

For neural networks with millions of parameters, this is impractical.

### Weakness 2: Non-Convex Functions

For non-convex functions:
- Hessian may be indefinite → Newton step may increase cost
- Hessian may be singular → no solution
- May converge to saddle point or maximum

### Remedies

**Damped Newton** (add step size):
$$w^{(k+1)} = w^{(k)} - \alpha_k H^{-1} \nabla g$$

**Modified Hessian** (ensure positive definiteness):
$$\tilde{H} = H + \lambda I$$

**Line search**: Find $\alpha$ that satisfies Armijo condition.

---

## Quasi-Newton Methods

### Idea

Approximate the Hessian (or its inverse) using gradient information.

### BFGS Update

Maintain approximation $B_k \approx H_k$:

$$B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

Where:
- $s_k = w^{(k+1)} - w^{(k)}$
- $y_k = \nabla g(w^{(k+1)}) - \nabla g(w^{(k)})$

### L-BFGS

Limited-memory BFGS: Store only last $m$ updates (typically $m = 10$).
- Memory: $O(mn)$ instead of $O(n^2)$
- Practical for large-scale optimization

---

## Implementation

### Full Newton's Method

```python
import numpy as np
from numpy.linalg import solve, norm

def newtons_method(objective, gradient, hessian, x0, 
                   max_iter=100, tol=1e-8, damped=False):
    x = np.array(x0, dtype=float)
    history = {'x': [x.copy()], 'f': [objective(x)]}
    
    for k in range(max_iter):
        grad = gradient(x)
        H = hessian(x)
        
        # Check convergence
        if norm(grad) < tol:
            break
        
        # Newton direction
        try:
            d = solve(H, -grad)
        except np.linalg.LinAlgError:
            # Regularize if singular
            d = solve(H + 0.1*np.eye(len(x)), -grad)
        
        # Damped Newton with backtracking
        if damped:
            alpha = 1.0
            while objective(x + alpha*d) > objective(x) + 1e-4*alpha*grad@d:
                alpha *= 0.5
            x = x + alpha * d
        else:
            x = x + d
        
        history['x'].append(x.copy())
        history['f'].append(objective(x))
    
    return x, history

# Example: Minimize Rosenbrock
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    return np.array([
        [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
        [-400*x[0], 200]
    ])

x_opt, hist = newtons_method(rosenbrock, rosenbrock_grad, rosenbrock_hess,
                             [-1.0, 1.0], damped=True)
print(f"Optimum: {x_opt}")
```

### Using SciPy

```python
from scipy.optimize import minimize

result = minimize(rosenbrock, x0=[-1, 1], method='Newton-CG',
                  jac=rosenbrock_grad, hess=rosenbrock_hess)
print(result.x)
```

---

## Exercises

### Exercise 4.1 (Section 4.1)
For $g(w_1, w_2) = w_1^2 - w_2^2$, find all stationary points and classify them using the Hessian.

### Exercise 4.2 (Section 4.3)
Apply Newton's method to minimize $g(w) = w^4 - 3w^2 + 2$. Find all convergence points depending on initialization.

### Exercise 4.3 (Section 4.4)
Implement damped Newton with backtracking line search for the Rosenbrock function.

---

## Summary

- Second-order condition: $\nabla g = 0$ and $H \succ 0$
- Newton: $w \leftarrow w - H^{-1} \nabla g$
- Quadratic convergence near optimum
- Expensive and problematic for non-convex functions

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 4**.

### Supplementary Reading
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Chapters 3, 6.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 9.

---

## Next Week Preview

**Week 5: Linear Regression** (Chapter 5)
- Least squares linear regression
- Least absolute deviations
- Regression quality metrics
- Weighted regression
