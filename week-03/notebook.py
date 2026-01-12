import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Week 3: First-Order Optimization - Gradient Descent

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapter 3**

    ---

    ## Learning Objectives

    - Understand the first-order optimality condition
    - Derive and implement gradient descent
    - Explore learning rate selection
    - Identify weaknesses of gradient descent
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## The First-Order Optimality Condition (Section 3.2)

    For a differentiable function $g(w)$, a necessary condition for $w^*$ to be a local minimum:

    $$\nabla g(w^*) = 0$$

    ### Intuition

    - Gradient $\nabla g(w)$ points in direction of steepest **ascent**
    - At a minimum, there's no direction of descent
    - Hence gradient must be zero
    """)
    return


@app.cell
def _(np, plt):
    # Visualize gradient and optimality
    x = np.linspace(-3, 3, 200)
    g = x**2 + 1
    grad_g = 2*x
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Function
    ax1 = axes[0]
    ax1.plot(x, g, 'b-', linewidth=2)
    ax1.plot(0, 1, 'r*', markersize=15, label='Minimum at w=0')
    ax1.set_xlabel('w', fontsize=12)
    ax1.set_ylabel('g(w)', fontsize=12)
    ax1.set_title('Function g(w) = w² + 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient
    ax2 = axes[1]
    ax2.plot(x, grad_g, 'g-', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.plot(0, 0, 'r*', markersize=15, label='∇g(0) = 0')
    ax2.set_xlabel('w', fontsize=12)
    ax2.set_ylabel('∇g(w)', fontsize=12)
    ax2.set_title('Gradient ∇g(w) = 2w')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Geometry of First-Order Taylor Series (Section 3.3)

    Near a point $w$, the function can be approximated:

    $$g(w + d) \approx g(w) + \nabla g(w)^T d$$

    ### Key Insight

    The direction of **steepest descent** is:

    $$d = -\nabla g(w)$$

    Because $\nabla g(w)^T (-\nabla g(w)) = -\|\nabla g(w)\|^2 < 0$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient Descent (Section 3.5)

    ### The Algorithm

    $$w^{(k+1)} = w^{(k)} - \alpha \nabla g(w^{(k)})$$

    Where $\alpha > 0$ is the **learning rate** (step size).

    ### Pseudocode

    ```python
    def gradient_descent(g, grad_g, w0, alpha, max_iter):
        w = w0
        for k in range(max_iter):
            w = w - alpha * grad_g(w)
            if converged(w):
                break
        return w
    ```
    """)
    return


@app.cell
def _(np, plt):
    # Gradient descent visualization
    def f(w):
        return w[0]**2 + 5*w[1]**2
    
    def grad_f(w):
        return np.array([2*w[0], 10*w[1]])
    
    def gradient_descent(grad_f, w0, alpha, n_iter):
        path = [w0.copy()]
        w = w0.copy()
        for _ in range(n_iter):
            w = w - alpha * grad_f(w)
            path.append(w.copy())
        return np.array(path)
    
    # Run gradient descent
    w0 = np.array([4.0, 2.0])
    path = gradient_descent(grad_f, w0, alpha=0.1, n_iter=20)
    
    # Contour plot
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 5*Y**2
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.contour(X, Y, Z, levels=30, cmap='viridis')
    ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=8, linewidth=2, label='GD Path')
    ax2.plot(path[0, 0], path[0, 1], 'g*', markersize=15, label='Start')
    ax2.plot(0, 0, 'b*', markersize=15, label='Optimum')
    
    ax2.set_xlabel('$w_1$', fontsize=12)
    ax2.set_ylabel('$w_2$', fontsize=12)
    ax2.set_title('Gradient Descent on f(w) = $w_1^2$ + 5$w_2^2$ (ML Refined, Section 3.5)', fontsize=14)
    ax2.legend()
    ax2.set_aspect('equal')
    
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Learning Rate Selection

    The learning rate $\alpha$ is crucial:

    | $\alpha$ | Effect |
    |----------|--------|
    | Too small | Very slow convergence |
    | Just right | Smooth, efficient convergence |
    | Too large | Oscillation |
    | Very large | Divergence |
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="Learning Rate α")
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(alpha_slider, np, plt):
    # Interactive learning rate demo
    def f_demo(w):
        return w**2
    
    def grad_demo(w):
        return 2*w
    
    alpha = alpha_slider.value
    w = 4.0
    history = [w]
    
    for _ in range(30):
        w = w - alpha * grad_demo(w)
        history.append(w)
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Path in function space
    ax1 = axes3[0]
    x = np.linspace(-5, 5, 100)
    ax1.plot(x, x**2, 'b-', linewidth=2)
    ax1.plot(history, [h**2 for h in history], 'ro-', markersize=6)
    ax1.set_xlabel('w')
    ax1.set_ylabel('g(w)')
    ax1.set_title(f'GD Path with α = {alpha}')
    ax1.grid(True, alpha=0.3)
    
    # Convergence
    ax2 = axes3[1]
    ax2.plot(range(len(history)), [h**2 for h in history], 'b-o', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('g(w)')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log' if min([h**2 for h in history]) > 0 else 'linear')
    
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Two Natural Weaknesses of Gradient Descent (Section 3.6)

    ### Weakness 1: Zigzagging

    For ill-conditioned functions (very different curvature in different directions),
    gradient descent zigzags inefficiently.

    ### Weakness 2: Saddle Points

    In high dimensions, saddle points are common. Gradient is zero but it's not a minimum!

    ### Solutions (Covered in Appendix A)

    - Momentum
    - Adaptive learning rates (Adam, RMSprop)
    - Second-order methods
    """)
    return


@app.cell
def _(np, plt):
    # Saddle point visualization
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 - Y**2  # Saddle function
    
    fig4 = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig4.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.plot([0], [0], [0], 'r*', markersize=15)
    ax1.set_xlabel('$w_1$')
    ax1.set_ylabel('$w_2$')
    ax1.set_zlabel('g(w)')
    ax1.set_title('Saddle Point at (0, 0)')
    
    # Contour
    ax2 = fig4.add_subplot(122)
    ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.plot(0, 0, 'r*', markersize=15, label='Saddle point: ∇g = 0')
    ax2.set_xlabel('$w_1$')
    ax2.set_ylabel('$w_2$')
    ax2.set_title('g(w) = $w_1^2$ - $w_2^2$')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Key Points |
    |---------|------------|
    | **First-order condition** | $\nabla g(w^*) = 0$ at stationary points |
    | **Gradient descent** | $w^{(k+1)} = w^{(k)} - \alpha \nabla g(w^{(k)})$ |
    | **Learning rate** | Critical for convergence |
    | **Weaknesses** | Zigzagging, saddle points |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapter 3.
    - **Supplementary**: Nocedal, J. & Wright, S. (2006). *Numerical Optimization*, Chapter 3.

    ## Next Week

    **Second-Order Optimization: Newton's Method** (Chapter 4): Using curvature information for faster convergence.
    """)
    return


if __name__ == "__main__":
    app.run()
