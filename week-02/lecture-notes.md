# Week 2: Zero-Order Optimization Techniques

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 2: Zero-Order Optimization Techniques**.

---

## Overview

This week covers optimization methods that rely only on function evaluations—no derivatives required.

---

## Learning Objectives

- Understand the zero-order optimality condition
- Apply global optimization methods (grid search)
- Apply local optimization methods
- Implement random search and coordinate descent

---

## 2.1 Introduction

### What are Zero-Order Methods?

Optimization algorithms that use only **function values** $g(w)$, not derivatives.

### When to Use

- Derivative doesn't exist or is discontinuous
- Derivative is expensive to compute
- Black-box optimization (simulation, hardware)
- Hyperparameter tuning in ML

---

## 2.2 The Zero-Order Optimality Condition

### Global Minimum

A point $w^*$ is a **global minimum** of $g$ if:
$$g(w^*) \leq g(w) \quad \forall w \in \mathbb{R}^n$$

### Local Minimum

A point $w^*$ is a **local minimum** if there exists $\epsilon > 0$ such that:
$$g(w^*) \leq g(w) \quad \forall w : \|w - w^*\| < \epsilon$$

### Key Insight

Without derivative information, we can only compare function values at sampled points.

---

## 2.3 Global Optimization Methods

### Exhaustive Grid Search

**Algorithm:**
1. Define a grid of points over the search space
2. Evaluate $g(w)$ at every grid point
3. Return the point with minimum value

**Pseudocode:**
```python
def grid_search(g, bounds, n_points):
    grid = create_grid(bounds, n_points)
    best_w = None
    best_val = float('inf')
    
    for w in grid:
        val = g(w)
        if val < best_val:
            best_val = val
            best_w = w
    
    return best_w
```

### Computational Complexity

For $N$ points per dimension and $d$ dimensions:
$$\text{Total evaluations} = N^d$$

**Example:**
- $N = 100$, $d = 10$: $100^{10} = 10^{20}$ evaluations!

This is the **curse of dimensionality**.

---

## 2.4 Local Optimization Methods

### The General Descent Framework

Starting from initial point $w^{(0)}$, iterate:

$$w^{(k+1)} = w^{(k)} + \alpha_k d^{(k)}$$

Where:
- $d^{(k)}$: Descent direction
- $\alpha_k$: Step size (learning rate)

### Requirements for Descent

A direction $d$ is a descent direction at $w$ if:
$$g(w + \alpha d) < g(w) \quad \text{for small } \alpha > 0$$

### Local vs Global

Local methods find local minima—no guarantee of global optimum.

**Mitigation strategies:**
- Multiple random restarts
- Combine with global search
- Simulated annealing

---

## 2.5 Random Search

### Algorithm

Sample points uniformly at random and keep the best:

```python
def random_search(g, bounds, n_samples):
    best_w = None
    best_val = float('inf')
    
    for k in range(n_samples):
        w = sample_uniform(bounds)
        val = g(w)
        if val < best_val:
            best_val = val
            best_w = w
    
    return best_w
```

### Properties

| Aspect | Characteristic |
|--------|---------------|
| Simplicity | Very easy to implement |
| Parallelization | Embarrassingly parallel |
| Global | Can find global minimum (probabilistically) |
| Efficiency | Slow for high dimensions |

### Comparison with Grid Search

For the same number of evaluations:
- Grid Search: Systematic coverage, may miss between grid points
- Random Search: Better exploration, especially for high dimensions

### Random Search for Hyperparameters

Bergstra & Bengio (2012) showed random search often outperforms grid search for hyperparameter optimization because:
- Not all hyperparameters are equally important
- Random search explores important dimensions more efficiently

---

## 2.6 Coordinate Search and Descent

### Coordinate Search

Optimize one variable at a time:

$$w_j^{(k+1)} = \arg\min_{w_j} g(w_1^{(k+1)}, \ldots, w_{j-1}^{(k+1)}, w_j, w_{j+1}^{(k)}, \ldots, w_n^{(k)})$$

**Algorithm:**
```python
def coordinate_search(g, w0, n_iter):
    w = w0.copy()
    n = len(w)
    
    for k in range(n_iter):
        for j in range(n):
            # Optimize coordinate j
            w[j] = minimize_1d(lambda wj: g_with_fixed(w, j, wj))
    
    return w
```

### Coordinate Descent

Move along coordinate directions using line search:

```python
def coordinate_descent(g, w0, n_iter, step):
    w = w0.copy()
    n = len(w)
    
    for k in range(n_iter):
        for j in range(n):
            # Line search along coordinate j
            alpha = line_search(g, w, e_j)
            w[j] = w[j] + alpha
    
    return w
```

### Convergence Properties

For convex functions:
- Coordinate descent converges to global minimum
- Rate depends on condition number

For non-convex functions:
- May converge to local minimum
- Can get stuck if coordinates are coupled

### Advantages

1. No gradient computation needed
2. Simple to implement
3. Each subproblem is 1-dimensional
4. Works well for separable functions

### Disadvantages

1. Can have zigzag behavior
2. Slow for highly correlated variables
3. Order of coordinates can matter

---

## Implementation Examples

### Random Search in Python

```python
import numpy as np

def random_search(objective, bounds, n_samples=1000, seed=42):
    np.random.seed(seed)
    n_dims = len(bounds)
    
    best_x = None
    best_val = np.inf
    
    for _ in range(n_samples):
        x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        val = objective(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val

# Example: Minimize Rosenbrock function
def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

bounds = [(-5, 5), (-5, 5)]
x_opt, f_opt = random_search(rosenbrock, bounds, n_samples=10000)
print(f"Best x: {x_opt}, f(x): {f_opt}")
```

### Coordinate Descent in Python

```python
import numpy as np
from scipy.optimize import minimize_scalar

def coordinate_descent(objective, x0, n_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    n = len(x)
    
    for k in range(n_iter):
        x_old = x.copy()
        
        for j in range(n):
            # Create 1D function
            def f_1d(xj):
                x_temp = x.copy()
                x_temp[j] = xj
                return objective(x_temp)
            
            # Minimize along coordinate j
            result = minimize_scalar(f_1d)
            x[j] = result.x
        
        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x

# Example
def quadratic(x):
    return x[0]**2 + 10*x[1]**2

x_opt = coordinate_descent(quadratic, [5.0, 3.0])
print(f"Optimum: {x_opt}")
```

---

## Exercises

### Exercise 2.1 (Section 2.3)
For a 5-dimensional problem with 20 grid points per dimension, how many function evaluations does grid search require?

### Exercise 2.2 (Section 2.5)
Implement random search to minimize the Rastrigin function:
$$f(x) = 10n + \sum_{i=1}^n [x_i^2 - 10\cos(2\pi x_i)]$$

### Exercise 2.3 (Section 2.6)
Apply coordinate descent to minimize $f(x,y) = x^2 + y^2 + xy$ starting from $(3, 3)$. Show the first 5 iterations.

---

## Summary

- Zero-order methods use only function evaluations
- Grid search is exhaustive but exponentially expensive
- Random search is simple and often effective
- Coordinate descent optimizes one variable at a time

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 2**.

### Supplementary Reading
- Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR*, 13, 281-305.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Chapter 9.

---

## Next Week Preview

**Week 3: First-Order Optimization: Gradient Descent** (Chapter 3)
- First-order optimality conditions
- Gradient descent algorithm
- Learning rate selection
- Convergence properties
