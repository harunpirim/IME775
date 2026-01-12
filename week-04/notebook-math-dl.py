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
    # Week 4: Neural Network Foundations - Perceptrons to MLPs

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 5

    ---

    ## Learning Objectives

    - Understand the perceptron as a linear classifier
    - Master multi-layer network mathematics
    - Learn forward propagation computation
    - Visualize decision boundaries
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    return ListedColormap, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.1 The Perceptron: A Single Neuron

    $$y = \sigma(w^T x + b) = \sigma\left(\sum_{i=1}^n w_i x_i + b\right)$$

    The perceptron defines a **hyperplane** that separates classes.
    """)
    return


@app.cell
def _(ListedColormap, np, plt):
    # Visualize single perceptron decision boundary
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Generate linearly separable data
    np.random.seed(42)
    n_samples = 100
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X_linear = np.vstack([X_class0, X_class1])
    y_linear = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Define perceptron weights (learned or set manually)
    w = np.array([1.0, 1.0])
    b = 0.0

    # Decision boundary: w^T x + b = 0  =>  x2 = -(w1*x1 + b)/w2
    x1_line = np.linspace(-5, 5, 100)
    x2_line = -(w[0] * x1_line + b) / w[1]

    # Plot
    ax1 = axes1[0]
    ax1.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='Class 0', alpha=0.6)
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='Class 1', alpha=0.6)
    ax1.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision boundary')

    # Add weight vector
    ax1.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, 
              color='green', width=0.03, label='Weight vector')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Perceptron: Linear Decision Boundary')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # XOR problem - not linearly separable
    ax2 = axes1[1]
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    colors_xor = ['blue' if y == 0 else 'red' for y in y_xor]
    ax2.scatter(X_xor[:, 0], X_xor[:, 1], c=colors_xor, s=200, edgecolors='black')

    for i, (x, y) in enumerate(zip(X_xor, y_xor)):
        ax2.annotate(f'XOR={y}', xy=(x[0], x[1]), xytext=(x[0]+0.1, x[1]+0.1), fontsize=12)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('XOR Problem: NOT Linearly Separable\n(Need multiple layers!)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1
    return (
        X_class0,
        X_class1,
        X_linear,
        X_xor,
        ax1,
        ax2,
        axes1,
        b,
        colors_xor,
        fig1,
        i,
        n_samples,
        w,
        x,
        x1_line,
        x2_line,
        y,
        y_linear,
        y_xor,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.2 Activation Functions

    | Function | Formula | Use Case |
    |----------|---------|----------|
    | Sigmoid | $\frac{1}{1+e^{-x}}$ | Output for binary classification |
    | ReLU | $\max(0, x)$ | Hidden layers (most common) |
    | Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Hidden layers (zero-centered) |
    | Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Multi-class output |
    """)
    return


@app.cell
def _(np, plt):
    # Visualize activation functions
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    x_act = np.linspace(-5, 5, 200)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x_act))
    axes2[0, 0].plot(x_act, sigmoid, 'b-', linewidth=2, label='σ(x)')
    axes2[0, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 0].fill_between(x_act, sigmoid, alpha=0.2)
    axes2[0, 0].set_title('Sigmoid: Output ∈ (0, 1)')
    axes2[0, 0].set_xlabel('x')
    axes2[0, 0].set_ylabel('σ(x)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)

    # Tanh
    tanh = np.tanh(x_act)
    axes2[0, 1].plot(x_act, tanh, 'g-', linewidth=2, label='tanh(x)')
    axes2[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[0, 1].fill_between(x_act, tanh, alpha=0.2, color='green')
    axes2[0, 1].set_title('Tanh: Output ∈ (-1, 1), Zero-Centered')
    axes2[0, 1].set_xlabel('x')
    axes2[0, 1].set_ylabel('tanh(x)')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    # ReLU
    relu = np.maximum(0, x_act)
    axes2[1, 0].plot(x_act, relu, 'r-', linewidth=2, label='ReLU(x)')
    axes2[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 0].fill_between(x_act, relu, alpha=0.2, color='red')
    axes2[1, 0].set_title('ReLU: Non-Saturating for x > 0')
    axes2[1, 0].set_xlabel('x')
    axes2[1, 0].set_ylabel('ReLU(x)')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    # Leaky ReLU and variants
    leaky_relu = np.where(x_act > 0, x_act, 0.1 * x_act)
    elu = np.where(x_act > 0, x_act, np.exp(x_act) - 1)
    axes2[1, 1].plot(x_act, relu, 'r-', linewidth=2, label='ReLU', alpha=0.5)
    axes2[1, 1].plot(x_act, leaky_relu, 'm-', linewidth=2, label='Leaky ReLU (α=0.1)')
    axes2[1, 1].plot(x_act, elu, 'c-', linewidth=2, label='ELU')
    axes2[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes2[1, 1].set_title('ReLU Variants: Avoid Dead Neurons')
    axes2[1, 1].set_xlabel('x')
    axes2[1, 1].set_ylabel('f(x)')
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2
    return axes2, elu, fig2, leaky_relu, relu, sigmoid, tanh, x_act


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.3 Multi-Layer Perceptron (MLP)

    **Forward Propagation**:
    
    For layer $l$:
    $$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
    $$h^{(l)} = \sigma(z^{(l)})$$

    Stacking layers enables learning complex non-linear functions!
    """)
    return


@app.cell
def _(np, plt):
    # Implement and visualize MLP solving XOR
    class SimpleMLP:
        def __init__(self, layer_sizes):
            np.random.seed(42)
            self.weights = []
            self.biases = []
            for i in range(len(layer_sizes) - 1):
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
                b = np.zeros(layer_sizes[i+1])
                self.weights.append(W)
                self.biases.append(b)
        
        def forward(self, X):
            self.activations = [X]
            h = X
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = h @ W + b
                if i < len(self.weights) - 1:
                    h = np.maximum(0, z)  # ReLU
                else:
                    h = 1 / (1 + np.exp(-z))  # Sigmoid output
                self.activations.append(h)
            return h
        
        def train(self, X, y, lr=0.1, epochs=1000):
            losses = []
            for epoch in range(epochs):
                # Forward
                pred = self.forward(X)
                
                # Loss (binary cross-entropy)
                loss = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
                losses.append(loss)
                
                # Backward (manual gradient computation)
                grad = pred - y.reshape(-1, 1)
                
                for i in range(len(self.weights) - 1, -1, -1):
                    dW = self.activations[i].T @ grad / len(X)
                    db = np.mean(grad, axis=0)
                    
                    if i > 0:
                        grad = grad @ self.weights[i].T
                        grad = grad * (self.activations[i] > 0)  # ReLU derivative
                    
                    self.weights[i] -= lr * dW
                    self.biases[i] -= lr * db
            
            return losses

    # Train on XOR
    X_xor_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor_train = np.array([0, 1, 1, 0])

    mlp = SimpleMLP([2, 8, 8, 1])  # 2 input, two hidden layers of 8, 1 output
    losses_xor = mlp.train(X_xor_train, y_xor_train, lr=1.0, epochs=2000)

    # Visualize
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes3[0].plot(losses_xor, 'b-', linewidth=1)
    axes3[0].set_xlabel('Epoch')
    axes3[0].set_ylabel('Loss')
    axes3[0].set_title('Training Loss: MLP Learning XOR')
    axes3[0].grid(True, alpha=0.3)

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_xor = mlp.forward(grid).reshape(xx.shape)

    axes3[1].contourf(xx, yy, Z_xor, levels=50, cmap='RdYlBu', alpha=0.7)
    axes3[1].contour(xx, yy, Z_xor, levels=[0.5], colors='black', linewidths=2)

    colors_xor_vis = ['blue' if y == 0 else 'red' for y in y_xor_train]
    axes3[1].scatter(X_xor_train[:, 0], X_xor_train[:, 1], c=colors_xor_vis, 
                     s=200, edgecolors='black', linewidth=2)

    for xi, yi, label in zip(X_xor_train[:, 0], X_xor_train[:, 1], y_xor_train):
        pred_val = mlp.forward(np.array([[xi, yi]]))[0, 0]
        axes3[1].annotate(f'pred={pred_val:.2f}', xy=(xi, yi), 
                         xytext=(xi+0.1, yi+0.1), fontsize=10)

    axes3[1].set_xlabel('x₁')
    axes3[1].set_ylabel('x₂')
    axes3[1].set_title('MLP Decision Boundary: XOR Solved!')
    axes3[1].set_xlim(-0.5, 1.5)
    axes3[1].set_ylim(-0.5, 1.5)

    plt.tight_layout()
    fig3
    return (
        SimpleMLP,
        X_xor_train,
        Z_xor,
        axes3,
        colors_xor_vis,
        fig3,
        grid,
        label,
        losses_xor,
        mlp,
        pred_val,
        xi,
        xx,
        yi,
        yy,
        y_xor_train,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.4 Universal Approximation

    **Theorem**: A single hidden layer network with sufficient neurons can approximate any continuous function.

    Let's visualize how more neurons improve approximation!
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate universal approximation
    def target_function(x):
        return np.sin(2 * x) + 0.5 * np.cos(5 * x)

    class SimpleApproximator:
        def __init__(self, hidden_size):
            np.random.seed(42)
            self.W1 = np.random.randn(1, hidden_size) * 2
            self.b1 = np.random.randn(hidden_size) * 2
            self.W2 = np.random.randn(hidden_size, 1) * 0.5
            self.b2 = np.zeros(1)
        
        def forward(self, x):
            h = np.tanh(x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
        
        def train(self, X, y, lr=0.01, epochs=2000):
            for _ in range(epochs):
                # Forward
                h = np.tanh(X @ self.W1 + self.b1)
                pred = h @ self.W2 + self.b2
                
                # Backward
                error = pred - y
                dW2 = h.T @ error / len(X)
                db2 = np.mean(error)
                
                dh = error @ self.W2.T
                dtanh = dh * (1 - h**2)
                dW1 = X.T @ dtanh / len(X)
                db1 = np.mean(dtanh, axis=0)
                
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

    # Train with different numbers of hidden neurons
    X_approx = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_approx = target_function(X_approx)

    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

    hidden_sizes = [2, 5, 20, 50]

    for ax, n_hidden in zip(axes4.flat, hidden_sizes):
        model = SimpleApproximator(n_hidden)
        model.train(X_approx, y_approx, lr=0.1, epochs=3000)
        
        y_pred = model.forward(X_approx)
        mse = np.mean((y_pred - y_approx)**2)
        
        ax.plot(X_approx, y_approx, 'b-', linewidth=2, label='Target')
        ax.plot(X_approx, y_pred, 'r--', linewidth=2, label='Approximation')
        ax.set_title(f'{n_hidden} Hidden Neurons (MSE: {mse:.4f})')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Universal Approximation: More Neurons → Better Fit', fontsize=14, y=1.02)
    plt.tight_layout()
    fig4
    return (
        SimpleApproximator,
        X_approx,
        ax,
        axes4,
        fig4,
        hidden_sizes,
        model,
        mse,
        n_hidden,
        target_function,
        y_approx,
        y_pred,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4.5 Network Architecture Visualization

    Understanding how dimensions flow through a network is crucial!
    """)
    return


@app.cell
def _(np, plt):
    # Visualize network architecture
    def draw_network(ax, layer_sizes, title):
        n_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        layer_positions = np.linspace(0, 1, n_layers)
        
        for i, (pos, n_neurons) in enumerate(zip(layer_positions, layer_sizes)):
            # Calculate vertical positions for neurons
            neuron_positions = np.linspace(0.1, 0.9, min(n_neurons, 10))
            
            for j, y_pos in enumerate(neuron_positions):
                circle = plt.Circle((pos, y_pos), 0.03, fill=True, 
                                   color='lightblue' if i == 0 else ('lightgreen' if i == n_layers-1 else 'lightyellow'),
                                   edgecolor='black', linewidth=1)
                ax.add_patch(circle)
            
            # Draw connections to next layer
            if i < n_layers - 1:
                next_positions = np.linspace(0.1, 0.9, min(layer_sizes[i+1], 10))
                for y1 in neuron_positions[:min(5, len(neuron_positions))]:
                    for y2 in next_positions[:min(5, len(next_positions))]:
                        ax.plot([pos, layer_positions[i+1]], [y1, y2], 
                               'gray', linewidth=0.3, alpha=0.3)
            
            # Add label
            ax.text(pos, -0.05, f'Layer {i}\n({layer_sizes[i]})', ha='center', fontsize=10)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, pad=10)

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

    # Architecture 1: Wide and shallow
    draw_network(axes5[0], [784, 512, 10], 'Shallow: 784 → 512 → 10\nParams: ~400K')

    # Architecture 2: Deep and narrow  
    draw_network(axes5[1], [784, 256, 128, 64, 10], 'Deep: 784 → 256 → 128 → 64 → 10\nParams: ~240K')

    plt.tight_layout()
    fig5
    return axes5, draw_network, fig5


@app.cell
def _(np):
    # Parameter counting
    def count_params(layer_sizes):
        total = 0
        for i in range(len(layer_sizes) - 1):
            weights = layer_sizes[i] * layer_sizes[i+1]
            biases = layer_sizes[i+1]
            total += weights + biases
        return total

    architectures = {
        'Shallow [784, 512, 10]': [784, 512, 10],
        'Deep [784, 256, 128, 64, 10]': [784, 256, 128, 64, 10],
        'MNIST typical [784, 128, 64, 10]': [784, 128, 64, 10]
    }

    print("Parameter Counts:")
    print("-" * 50)
    for name, layers in architectures.items():
        params = count_params(layers)
        print(f"{name}: {params:,} parameters")
    return architectures, count_params, layers, name, params


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Key Insight |
    |---------|-------------|
    | **Perceptron** | Linear classifier, limited to linearly separable data |
    | **Activation** | Non-linearity enables complex function learning |
    | **MLP** | Stacked layers learn hierarchical representations |
    | **Universal Approximation** | Sufficient neurons can approximate any function |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 5.
    - **Supplementary**: Goodfellow, I., et al. (2016). *Deep Learning*, Chapter 6.

    ## Connection to ML Refined Curriculum

    Neural networks extend the linear models from Weeks 4-7 to handle non-linear patterns:
    - Linear regression → Neural network regression
    - Logistic regression → Neural network classification
    """)
    return


if __name__ == "__main__":
    app.run()

