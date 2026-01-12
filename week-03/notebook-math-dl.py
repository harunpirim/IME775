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
    # Week 3: Gradient-Based Optimization for Deep Learning

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 4

    ---

    ## Learning Objectives

    - Understand gradient descent and its variants
    - Master momentum-based acceleration
    - Learn adaptive learning rate methods (Adam)
    - Connect optimization to neural network training
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    return Axes3D, cm, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.1 Gradient Descent Visualization

    Basic update: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$

    The gradient points toward steepest ascent, so we go the opposite way.
    """)
    return


@app.cell
def _(np, plt):
    # Visualize gradient descent on a 2D function
    def rosenbrock(x, y, a=1, b=100):
        return (a - x)**2 + b * (y - x**2)**2

    def rosenbrock_grad(x, y, a=1, b=100):
        dx = -2*(a - x) - 4*b*x*(y - x**2)
        dy = 2*b*(y - x**2)
        return np.array([dx, dy])

    def gradient_descent(grad_f, x0, lr=0.001, n_iters=100):
        path = [x0.copy()]
        x = x0.copy()
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    # Create contour plot
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)

    # Different learning rates
    learning_rates = [0.001, 0.003]
    colors = ['blue', 'red']
    start = np.array([-1.5, 2.0])

    for ax, lr, color in zip(axes1, learning_rates, colors):
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
        
        path = gradient_descent(rosenbrock_grad, start, lr=lr, n_iters=500)
        ax.plot(path[:, 0], path[:, 1], f'{color}.-', markersize=2, linewidth=0.5, alpha=0.7)
        ax.scatter(path[0, 0], path[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)
        ax.scatter(1, 1, color='gold', s=150, marker='★', label='Optimum', zorder=5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Gradient Descent (lr={lr})\nIterations: 500, Final loss: {rosenbrock(path[-1,0], path[-1,1]):.2f}')
        ax.legend()

    plt.tight_layout()
    fig1
    return (
        X,
        Y,
        Z,
        ax,
        axes1,
        color,
        colors,
        fig1,
        gradient_descent,
        learning_rates,
        lr,
        path,
        rosenbrock,
        rosenbrock_grad,
        start,
        x_range,
        y_range,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.2 Momentum: Accelerating Convergence

    Momentum accumulates velocity in consistent gradient directions:

    $$v_{t+1} = \beta v_t + \nabla L(\theta_t)$$
    $$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

    Like a ball rolling down a hill - accelerates in consistent directions!
    """)
    return


@app.cell
def _(np, plt, rosenbrock, rosenbrock_grad):
    def gd_momentum(grad_f, x0, lr=0.001, momentum=0.9, n_iters=100):
        path = [x0.copy()]
        x = x0.copy()
        v = np.zeros_like(x)
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            v = momentum * v + g
            x = x - lr * v
            path.append(x.copy())
        return np.array(path)

    # Compare GD vs Momentum on ill-conditioned problem
    def ill_conditioned(x, y):
        return x**2 + 50*y**2

    def ill_conditioned_grad(x, y):
        return np.array([2*x, 100*y])

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    x_ill = np.linspace(-5, 5, 100)
    y_ill = np.linspace(-1, 1, 100)
    X_ill, Y_ill = np.meshgrid(x_ill, y_ill)
    Z_ill = ill_conditioned(X_ill, Y_ill)

    start_ill = np.array([4.0, 0.8])
    n_iters_ill = 100

    # Vanilla GD
    def gd_simple(grad_f, x0, lr=0.01, n_iters=100):
        path = [x0.copy()]
        x = x0.copy()
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    path_gd = gd_simple(ill_conditioned_grad, start_ill, lr=0.015, n_iters=n_iters_ill)
    path_mom = gd_momentum(ill_conditioned_grad, start_ill, lr=0.015, momentum=0.9, n_iters=n_iters_ill)

    # Plot GD
    axes2[0].contour(X_ill, Y_ill, Z_ill, levels=20, cmap='viridis')
    axes2[0].plot(path_gd[:, 0], path_gd[:, 1], 'b.-', markersize=3, linewidth=0.5, alpha=0.7)
    axes2[0].scatter(0, 0, color='gold', s=150, marker='★', zorder=5)
    axes2[0].set_xlabel('x')
    axes2[0].set_ylabel('y')
    axes2[0].set_title(f'Vanilla GD (Oscillates)\nFinal loss: {ill_conditioned(path_gd[-1,0], path_gd[-1,1]):.4f}')

    # Plot Momentum
    axes2[1].contour(X_ill, Y_ill, Z_ill, levels=20, cmap='viridis')
    axes2[1].plot(path_mom[:, 0], path_mom[:, 1], 'r.-', markersize=3, linewidth=0.5, alpha=0.7)
    axes2[1].scatter(0, 0, color='gold', s=150, marker='★', zorder=5)
    axes2[1].set_xlabel('x')
    axes2[1].set_ylabel('y')
    axes2[1].set_title(f'GD + Momentum (Smoother)\nFinal loss: {ill_conditioned(path_mom[-1,0], path_mom[-1,1]):.4f}')

    plt.tight_layout()
    fig2
    return (
        X_ill,
        Y_ill,
        Z_ill,
        axes2,
        fig2,
        gd_momentum,
        gd_simple,
        ill_conditioned,
        ill_conditioned_grad,
        n_iters_ill,
        path_gd,
        path_mom,
        start_ill,
        x_ill,
        y_ill,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.3 Adam: Adaptive Moment Estimation

    Combines momentum with adaptive learning rates:

    **First moment** (momentum): $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$

    **Second moment** (RMSprop): $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

    **Update**: $\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
    """)
    return


@app.cell
def _(np, plt, rosenbrock, rosenbrock_grad):
    class AdamOptimizer:
        def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.m = None
            self.v = None
            self.t = 0
        
        def step(self, x, grad):
            if self.m is None:
                self.m = np.zeros_like(x)
                self.v = np.zeros_like(x)
            
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def run_adam(grad_f, x0, lr=0.1, n_iters=100):
        adam = AdamOptimizer(lr=lr)
        path = [x0.copy()]
        x = x0.copy()
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            x = adam.step(x, g)
            path.append(x.copy())
        return np.array(path)

    # Compare optimizers on Rosenbrock
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

    x_r = np.linspace(-2, 2, 100)
    y_r = np.linspace(-1, 3, 100)
    X_r, Y_r = np.meshgrid(x_r, y_r)
    Z_r = rosenbrock(X_r, Y_r)

    start_r = np.array([-1.0, 2.0])
    n_iters_r = 300

    # Run each optimizer
    def gd_simple2(grad_f, x0, lr=0.001, n_iters=100):
        path = [x0.copy()]
        x = x0.copy()
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    def gd_momentum2(grad_f, x0, lr=0.001, momentum=0.9, n_iters=100):
        path = [x0.copy()]
        x = x0.copy()
        v = np.zeros_like(x)
        for _ in range(n_iters):
            g = grad_f(x[0], x[1])
            v = momentum * v + g
            x = x - lr * v
            path.append(x.copy())
        return np.array(path)

    paths_compare = {
        'SGD': gd_simple2(rosenbrock_grad, start_r, lr=0.001, n_iters=n_iters_r),
        'Momentum': gd_momentum2(rosenbrock_grad, start_r, lr=0.001, momentum=0.9, n_iters=n_iters_r),
        'Adam': run_adam(rosenbrock_grad, start_r, lr=0.05, n_iters=n_iters_r)
    }

    colors_opt = ['blue', 'green', 'red']

    for ax, (name, path_opt), color in zip(axes3, paths_compare.items(), colors_opt):
        ax.contour(X_r, Y_r, Z_r, levels=np.logspace(-1, 3, 15), cmap='viridis')
        ax.plot(path_opt[:, 0], path_opt[:, 1], f'{color}.-', markersize=2, linewidth=0.5, alpha=0.8)
        ax.scatter(1, 1, color='gold', s=150, marker='★', zorder=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        final_loss = rosenbrock(path_opt[-1,0], path_opt[-1,1])
        ax.set_title(f'{name}\nFinal loss: {final_loss:.4f}')

    plt.tight_layout()
    fig3
    return (
        AdamOptimizer,
        X_r,
        Y_r,
        Z_r,
        ax,
        axes3,
        color,
        colors_opt,
        fig3,
        final_loss,
        gd_momentum2,
        gd_simple2,
        n_iters_r,
        name,
        path_opt,
        paths_compare,
        run_adam,
        start_r,
        x_r,
        y_r,
    )


@app.cell
def _(np, paths_compare, plt, rosenbrock):
    # Loss curves comparison
    fig4, ax4 = plt.subplots(figsize=(10, 5))

    for name, path in paths_compare.items():
        losses = [rosenbrock(p[0], p[1]) for p in path]
        ax4.semilogy(losses, label=name, linewidth=2)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Convergence Comparison on Rosenbrock Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig4
    return ax4, fig4, losses, name, path


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.4 Learning Rate Schedules

    | Schedule | Formula | Use Case |
    |----------|---------|----------|
    | Step Decay | $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | Classic CNNs |
    | Exponential | $\alpha_t = \alpha_0 \cdot e^{-\lambda t}$ | Smooth decay |
    | Cosine | $\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T}\pi))$ | Transformers |
    | Warmup | Linear increase then decay | Large batch training |
    """)
    return


@app.cell
def _(np, plt):
    # Visualize learning rate schedules
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 8))

    epochs = np.arange(100)
    alpha_0 = 0.1

    # Step decay
    step_lr = alpha_0 * (0.1 ** (epochs // 30))
    axes5[0, 0].plot(epochs, step_lr, 'b-', linewidth=2)
    axes5[0, 0].set_title('Step Decay (γ=0.1 every 30 epochs)')
    axes5[0, 0].set_xlabel('Epoch')
    axes5[0, 0].set_ylabel('Learning Rate')
    axes5[0, 0].grid(True, alpha=0.3)

    # Exponential decay
    exp_lr = alpha_0 * np.exp(-0.03 * epochs)
    axes5[0, 1].plot(epochs, exp_lr, 'g-', linewidth=2)
    axes5[0, 1].set_title('Exponential Decay (λ=0.03)')
    axes5[0, 1].set_xlabel('Epoch')
    axes5[0, 1].set_ylabel('Learning Rate')
    axes5[0, 1].grid(True, alpha=0.3)

    # Cosine annealing
    alpha_min = 0.001
    cosine_lr = alpha_min + 0.5 * (alpha_0 - alpha_min) * (1 + np.cos(np.pi * epochs / 100))
    axes5[1, 0].plot(epochs, cosine_lr, 'r-', linewidth=2)
    axes5[1, 0].set_title('Cosine Annealing')
    axes5[1, 0].set_xlabel('Epoch')
    axes5[1, 0].set_ylabel('Learning Rate')
    axes5[1, 0].grid(True, alpha=0.3)

    # Warmup + Cosine
    warmup_epochs = 10
    warmup_lr = np.where(epochs < warmup_epochs,
                         alpha_0 * epochs / warmup_epochs,
                         alpha_min + 0.5 * (alpha_0 - alpha_min) * (1 + np.cos(np.pi * (epochs - warmup_epochs) / (100 - warmup_epochs))))
    axes5[1, 1].plot(epochs, warmup_lr, 'm-', linewidth=2)
    axes5[1, 1].axvline(warmup_epochs, color='gray', linestyle='--', alpha=0.5, label='End warmup')
    axes5[1, 1].set_title('Warmup + Cosine Annealing')
    axes5[1, 1].set_xlabel('Epoch')
    axes5[1, 1].set_ylabel('Learning Rate')
    axes5[1, 1].legend()
    axes5[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig5
    return (
        alpha_0,
        alpha_min,
        axes5,
        cosine_lr,
        epochs,
        exp_lr,
        fig5,
        step_lr,
        warmup_epochs,
        warmup_lr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3.5 Weight Initialization

    | Method | Formula | Best For |
    |--------|---------|----------|
    | Xavier | $W \sim \mathcal{U}(-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})})$ | Sigmoid, Tanh |
    | He | $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$ | ReLU |
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate importance of initialization
    def forward_pass(W_list, x, activation='relu'):
        activations = [x]
        for W in W_list:
            x = x @ W
            if activation == 'relu':
                x = np.maximum(0, x)
            elif activation == 'tanh':
                x = np.tanh(x)
            activations.append(x)
        return activations

    np.random.seed(42)
    n_layers = 10
    hidden_size = 256
    batch_size = 32

    # Different initializations
    def create_network(init_type, n_layers, hidden_size):
        W_list = []
        for _ in range(n_layers):
            if init_type == 'small':
                W = np.random.randn(hidden_size, hidden_size) * 0.01
            elif init_type == 'large':
                W = np.random.randn(hidden_size, hidden_size) * 1.0
            elif init_type == 'xavier':
                W = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
            elif init_type == 'he':
                W = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
            W_list.append(W)
        return W_list

    fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))

    init_types = ['small', 'large', 'xavier', 'he']
    titles = ['Small Init (0.01)', 'Large Init (1.0)', 'Xavier Init', 'He Init']

    x_init = np.random.randn(batch_size, hidden_size)

    for ax, init_type, title in zip(axes6.flat, init_types, titles):
        W_list = create_network(init_type, n_layers, hidden_size)
        activations = forward_pass(W_list, x_init, activation='relu')
        
        # Plot activation statistics
        means = [np.mean(np.abs(a)) for a in activations]
        stds = [np.std(a) for a in activations]
        
        ax.semilogy(range(n_layers + 1), means, 'b.-', label='Mean |activation|', linewidth=2)
        ax.semilogy(range(n_layers + 1), stds, 'r.--', label='Std activation', linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Value (log scale)')
        ax.set_title(f'{title}\nFinal mean: {means[-1]:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6
    return (
        W_list,
        activations,
        ax,
        axes6,
        batch_size,
        create_network,
        fig6,
        forward_pass,
        hidden_size,
        init_type,
        init_types,
        means,
        n_layers,
        stds,
        title,
        titles,
        x_init,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Optimizer | Key Feature | When to Use |
    |-----------|-------------|-------------|
    | **SGD** | Simple, generalizes well | Final training |
    | **Momentum** | Accelerates convergence | Standard choice |
    | **Adam** | Adaptive + momentum | Quick prototyping |

    | Schedule | Key Feature | When to Use |
    |----------|-------------|-------------|
    | **Step Decay** | Simple, predictable | CNNs |
    | **Cosine** | Smooth, no hyperparameters | Modern networks |
    | **Warmup** | Stabilizes early training | Large models |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 4.
    - **Supplementary**: Ruder, S. "An overview of gradient descent optimization algorithms."

    ## Connection to ML Refined Curriculum

    These optimization techniques are used throughout:
    - Weeks 2-3: Foundation for all optimization
    - Weeks 4-13: Training any supervised learning model
    """)
    return


if __name__ == "__main__":
    app.run()

