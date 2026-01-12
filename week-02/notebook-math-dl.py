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
    # Week 2: Calculus Foundations for Deep Learning

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 3

    ---

    ## Learning Objectives

    - Master derivatives and their role in optimization
    - Understand multivariable calculus for neural networks
    - Learn the chain rule as the foundation of backpropagation
    - Connect gradients to learning algorithms
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return Axes3D, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.1 Derivatives: Measuring Change

    The derivative measures instantaneous rate of change:
    $$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

    **Geometric interpretation**: Slope of tangent line = direction of steepest ascent
    """)
    return


@app.cell
def _(np, plt):
    # Visualize derivative as slope
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Function and tangent
    ax1 = axes1[0]
    x = np.linspace(-2, 3, 200)
    f = lambda x: x**3 - 2*x**2 + 1
    f_prime = lambda x: 3*x**2 - 4*x

    ax1.plot(x, f(x), 'b-', linewidth=2, label='$f(x) = x^3 - 2x^2 + 1$')

    # Tangent at x=1.5
    x0 = 1.5
    slope = f_prime(x0)
    tangent = f(x0) + slope * (x - x0)
    ax1.plot(x, tangent, 'r--', linewidth=1.5, label=f'Tangent at x={x0}')
    ax1.scatter([x0], [f(x0)], color='red', s=100, zorder=5)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-3, 5)
    ax1.legend()
    ax1.set_title(f"Derivative: Slope = {slope:.2f}")
    ax1.grid(True, alpha=0.3)

    # Right: Derivative function
    ax2 = axes1[1]
    ax2.plot(x, f(x), 'b-', linewidth=2, label='$f(x)$', alpha=0.5)
    ax2.plot(x, f_prime(x), 'r-', linewidth=2, label="$f'(x) = 3x^2 - 4x$")
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)

    # Mark critical points (where f'(x) = 0)
    critical_x = [0, 4/3]
    for cx in critical_x:
        ax2.scatter([cx], [0], color='green', s=100, zorder=5, marker='o')
        ax2.scatter([cx], [f(cx)], color='green', s=100, zorder=5, marker='s')
        ax2.axvline(cx, color='green', linestyle=':', alpha=0.5)

    ax2.set_xlim(-2, 3)
    ax2.set_ylim(-3, 5)
    ax2.legend()
    ax2.set_title("Critical Points: $f'(x) = 0$")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1
    return (
        ax1,
        ax2,
        axes1,
        critical_x,
        cx,
        f,
        f_prime,
        fig1,
        slope,
        tangent,
        x,
        x0,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.2 The Gradient: Multivariate Extension

    For scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$:
    $$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$

    **Key property**: Gradient points toward steepest ascent!
    """)
    return


@app.cell
def _(np, plt):
    # Visualize gradient as direction of steepest ascent
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Define function
    f_2d = lambda x, y: x**2 + 2*y**2  # Elliptical paraboloid
    grad_f = lambda x, y: np.array([2*x, 4*y])

    # Create grid
    x_2d = np.linspace(-3, 3, 50)
    y_2d = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_2d, y_2d)
    Z = f_2d(X, Y)

    # Left: Contour plot with gradients
    ax1_grad = axes2[0]
    contour = ax1_grad.contour(X, Y, Z, levels=15, cmap='viridis')
    ax1_grad.clabel(contour, inline=True, fontsize=8)

    # Plot gradient vectors at selected points
    points = [(-2, -1.5), (-1, 1), (1.5, -1), (0.5, 1.5), (2, 0.5)]
    for px, py in points:
        g = grad_f(px, py)
        g_norm = g / np.linalg.norm(g) * 0.5  # Normalize for visualization
        ax1_grad.quiver(px, py, g_norm[0], g_norm[1], color='red', 
                       width=0.015, scale=5, label='Gradient' if (px, py) == points[0] else '')

    ax1_grad.set_xlabel('x')
    ax1_grad.set_ylabel('y')
    ax1_grad.set_title('Gradient: Direction of Steepest Ascent')
    ax1_grad.set_aspect('equal')

    # Right: 3D surface
    ax2_grad = axes2[1]
    ax2_grad = fig2.add_subplot(122, projection='3d')
    ax2_grad.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    ax2_grad.set_xlabel('x')
    ax2_grad.set_ylabel('y')
    ax2_grad.set_zlabel('f(x,y)')
    ax2_grad.set_title('$f(x,y) = x^2 + 2y^2$')

    plt.tight_layout()
    fig2
    return (
        X,
        Y,
        Z,
        ax1_grad,
        ax2_grad,
        axes2,
        contour,
        f_2d,
        fig2,
        g,
        g_norm,
        grad_f,
        points,
        px,
        py,
        x_2d,
        y_2d,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.3 Activation Functions and Their Derivatives

    | Function | Formula | Derivative |
    |----------|---------|------------|
    | Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
    | Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ |
    | ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ |
    | Leaky ReLU | $\max(\alpha x, x)$ | $\alpha$ or $1$ |
    """)
    return


@app.cell
def _(np, plt):
    # Activation functions and their derivatives
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

    x_act = np.linspace(-5, 5, 200)

    # Sigmoid
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sigmoid_grad = lambda x: sigmoid(x) * (1 - sigmoid(x))

    axes3[0, 0].plot(x_act, sigmoid(x_act), 'b-', linewidth=2, label='σ(x)')
    axes3[0, 0].plot(x_act, sigmoid_grad(x_act), 'r--', linewidth=2, label="σ'(x)")
    axes3[0, 0].axhline(0, color='gray', linewidth=0.5)
    axes3[0, 0].axvline(0, color='gray', linewidth=0.5)
    axes3[0, 0].legend()
    axes3[0, 0].set_title('Sigmoid: Saturates → Vanishing Gradients')
    axes3[0, 0].grid(True, alpha=0.3)

    # Tanh
    tanh_grad = lambda x: 1 - np.tanh(x)**2

    axes3[0, 1].plot(x_act, np.tanh(x_act), 'b-', linewidth=2, label='tanh(x)')
    axes3[0, 1].plot(x_act, tanh_grad(x_act), 'r--', linewidth=2, label="tanh'(x)")
    axes3[0, 1].axhline(0, color='gray', linewidth=0.5)
    axes3[0, 1].axvline(0, color='gray', linewidth=0.5)
    axes3[0, 1].legend()
    axes3[0, 1].set_title('Tanh: Zero-Centered but Still Saturates')
    axes3[0, 1].grid(True, alpha=0.3)

    # ReLU
    relu = lambda x: np.maximum(0, x)
    relu_grad = lambda x: (x > 0).astype(float)

    axes3[1, 0].plot(x_act, relu(x_act), 'b-', linewidth=2, label='ReLU(x)')
    axes3[1, 0].plot(x_act, relu_grad(x_act), 'r--', linewidth=2, label="ReLU'(x)")
    axes3[1, 0].axhline(0, color='gray', linewidth=0.5)
    axes3[1, 0].axvline(0, color='gray', linewidth=0.5)
    axes3[1, 0].legend()
    axes3[1, 0].set_title('ReLU: No Saturation (x > 0), Dead Neurons (x < 0)')
    axes3[1, 0].grid(True, alpha=0.3)

    # Softplus (smooth ReLU)
    softplus = lambda x: np.log(1 + np.exp(x))
    softplus_grad = sigmoid  # derivative of softplus is sigmoid!

    axes3[1, 1].plot(x_act, softplus(x_act), 'b-', linewidth=2, label='Softplus(x)')
    axes3[1, 1].plot(x_act, softplus_grad(x_act), 'r--', linewidth=2, label="Softplus'(x)")
    axes3[1, 1].plot(x_act, relu(x_act), 'g:', linewidth=1, alpha=0.5, label='ReLU (reference)')
    axes3[1, 1].axhline(0, color='gray', linewidth=0.5)
    axes3[1, 1].axvline(0, color='gray', linewidth=0.5)
    axes3[1, 1].legend()
    axes3[1, 1].set_title('Softplus: Smooth Approximation to ReLU')
    axes3[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3
    return (
        axes3,
        fig3,
        relu,
        relu_grad,
        sigmoid,
        sigmoid_grad,
        softplus,
        softplus_grad,
        tanh_grad,
        x_act,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.4 The Chain Rule: Foundation of Backpropagation

    For $z = f(g(x))$:
    $$\frac{dz}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

    **Neural network**: Each layer is a function composition
    $$L = \text{loss}(\text{layer}_n(\cdots \text{layer}_2(\text{layer}_1(x))))$$

    Gradients flow backward through the chain!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Example: Simple Neural Network

    Single neuron: $y = \sigma(wx + b)$

    Loss: $L = (y - t)^2$ where $t$ is target

    **Forward pass**:
    1. $z = wx + b$
    2. $y = \sigma(z)$
    3. $L = (y - t)^2$

    **Backward pass** (chain rule):
    - $\frac{\partial L}{\partial y} = 2(y - t)$
    - $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \cdot \sigma'(z)$
    - $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x$
    - $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate chain rule in action
    def forward_backward_demo():
        np.random.seed(42)
        
        # Simple neural network: y = sigmoid(wx + b)
        # Loss: L = (y - target)^2
        
        # Parameters
        w = 0.5
        b = 0.1
        x = 2.0
        target = 0.8
        
        # Forward pass
        z = w * x + b
        y = 1 / (1 + np.exp(-z))
        L = (y - target) ** 2
        
        # Backward pass (analytical gradients)
        dL_dy = 2 * (y - target)
        dy_dz = y * (1 - y)  # sigmoid derivative
        dL_dz = dL_dy * dy_dz
        dL_dw = dL_dz * x
        dL_db = dL_dz
        
        # Numerical gradients for verification
        eps = 1e-7
        
        # Numerical dL/dw
        z_plus = (w + eps) * x + b
        y_plus = 1 / (1 + np.exp(-z_plus))
        L_plus = (y_plus - target) ** 2
        z_minus = (w - eps) * x + b
        y_minus = 1 / (1 + np.exp(-z_minus))
        L_minus = (y_minus - target) ** 2
        numerical_dL_dw = (L_plus - L_minus) / (2 * eps)
        
        return {
            'forward': {'z': z, 'y': y, 'L': L},
            'analytical': {'dL_dw': dL_dw, 'dL_db': dL_db},
            'numerical': {'dL_dw': numerical_dL_dw}
        }

    result = forward_backward_demo()
    print("Forward Pass:")
    print(f"  z = {result['forward']['z']:.4f}")
    print(f"  y = {result['forward']['y']:.4f}")
    print(f"  L = {result['forward']['L']:.4f}")
    print("\nGradients (Chain Rule):")
    print(f"  Analytical dL/dw = {result['analytical']['dL_dw']:.6f}")
    print(f"  Numerical  dL/dw = {result['numerical']['dL_dw']:.6f}")
    print(f"  Match: {np.isclose(result['analytical']['dL_dw'], result['numerical']['dL_dw'])}")
    return forward_backward_demo, result


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.5 Gradient Checking

    **Numerical gradient** (for verification):
    $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

    **Gradient check**: Compare analytical vs numerical gradients

    This is crucial for debugging custom neural networks!
    """)
    return


@app.cell
def _(np, plt):
    # Gradient checking visualization
    def numerical_gradient(f, x, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad

    # Example: f(x) = x[0]^2 + 3*x[1]^2
    f_check = lambda x: x[0]**2 + 3*x[1]**2
    grad_f_analytical = lambda x: np.array([2*x[0], 6*x[1]])

    # Test points
    test_points = np.random.randn(20, 2)
    analytical_grads = np.array([grad_f_analytical(p) for p in test_points])
    numerical_grads = np.array([numerical_gradient(f_check, p) for p in test_points])

    # Visualize comparison
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    ax1_check = axes4[0]
    ax1_check.scatter(analytical_grads[:, 0], numerical_grads[:, 0], alpha=0.7, label='∂f/∂x₁')
    ax1_check.scatter(analytical_grads[:, 1], numerical_grads[:, 1], alpha=0.7, label='∂f/∂x₂')
    ax1_check.plot([-10, 10], [-10, 10], 'k--', alpha=0.5, label='Perfect match')
    ax1_check.set_xlabel('Analytical Gradient')
    ax1_check.set_ylabel('Numerical Gradient')
    ax1_check.set_title('Gradient Check: Analytical vs Numerical')
    ax1_check.legend()
    ax1_check.grid(True, alpha=0.3)

    # Relative error
    rel_errors = np.linalg.norm(analytical_grads - numerical_grads, axis=1) / (
        np.linalg.norm(analytical_grads, axis=1) + np.linalg.norm(numerical_grads, axis=1) + 1e-8)

    ax2_check = axes4[1]
    ax2_check.hist(rel_errors, bins=20, edgecolor='black', alpha=0.7)
    ax2_check.axvline(1e-7, color='r', linestyle='--', label='Typical threshold')
    ax2_check.set_xlabel('Relative Error')
    ax2_check.set_ylabel('Count')
    ax2_check.set_title(f'Relative Errors (max: {rel_errors.max():.2e})')
    ax2_check.legend()

    plt.tight_layout()
    fig4
    return (
        analytical_grads,
        ax1_check,
        ax2_check,
        axes4,
        f_check,
        fig4,
        grad_f_analytical,
        numerical_gradient,
        numerical_grads,
        rel_errors,
        test_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.6 The Hessian: Second-Order Information

    The Hessian matrix of second derivatives:
    $$\mathbf{H} = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2}
    \end{bmatrix}$$

    **Eigenvalues** tell us about curvature:
    - All positive → local minimum
    - All negative → local maximum
    - Mixed signs → saddle point
    """)
    return


@app.cell
def _(np, plt):
    # Visualize different curvature scenarios
    fig5 = plt.figure(figsize=(15, 4))

    x_hess = np.linspace(-2, 2, 50)
    y_hess = np.linspace(-2, 2, 50)
    X_hess, Y_hess = np.meshgrid(x_hess, y_hess)

    # Local minimum: f = x^2 + y^2
    ax1_hess = fig5.add_subplot(131, projection='3d')
    Z1 = X_hess**2 + Y_hess**2
    ax1_hess.plot_surface(X_hess, Y_hess, Z1, cmap='viridis', alpha=0.8)
    ax1_hess.set_title('Local Min: H positive definite\nEigenvalues: 2, 2')
    ax1_hess.set_xlabel('x')
    ax1_hess.set_ylabel('y')

    # Saddle point: f = x^2 - y^2
    ax2_hess = fig5.add_subplot(132, projection='3d')
    Z2 = X_hess**2 - Y_hess**2
    ax2_hess.plot_surface(X_hess, Y_hess, Z2, cmap='RdYlGn', alpha=0.8)
    ax2_hess.set_title('Saddle Point: H indefinite\nEigenvalues: 2, -2')
    ax2_hess.set_xlabel('x')
    ax2_hess.set_ylabel('y')

    # Ill-conditioned: f = x^2 + 100*y^2
    ax3_hess = fig5.add_subplot(133, projection='3d')
    Z3 = X_hess**2 + 10*Y_hess**2
    ax3_hess.plot_surface(X_hess, Y_hess, Z3, cmap='plasma', alpha=0.8)
    ax3_hess.set_title('Ill-conditioned: κ = 10\nEigenvalues: 2, 20')
    ax3_hess.set_xlabel('x')
    ax3_hess.set_ylabel('y')

    plt.tight_layout()
    fig5
    return (
        X_hess,
        Y_hess,
        Z1,
        Z2,
        Z3,
        ax1_hess,
        ax2_hess,
        ax3_hess,
        fig5,
        x_hess,
        y_hess,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Definition | Deep Learning Role |
    |---------|------------|-------------------|
    | **Derivative** | Rate of change | Sensitivity of loss to parameters |
    | **Gradient** | Vector of partials | Direction for parameter updates |
    | **Chain Rule** | Composition derivative | Backpropagation algorithm |
    | **Hessian** | Second derivatives | Curvature, adaptive learning rates |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 3.
    - **Supplementary**: Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 4.

    ## Connection to ML Refined Curriculum

    This calculus foundation supports:
    - Week 2-3: Optimization algorithms (gradient descent variants)
    - Weeks 4-8: Computing gradients for regression and classification
    """)
    return


if __name__ == "__main__":
    app.run()

