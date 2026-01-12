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
    # Week 5: Backpropagation - The Engine of Deep Learning

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 6

    ---

    ## Learning Objectives

    - Understand backpropagation as reverse-mode automatic differentiation
    - Derive gradients for common layers
    - Implement backpropagation from scratch
    - Verify gradients numerically
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
    ## 5.1 Why Backpropagation?

    **Problem**: Computing gradients for millions of parameters

    **Naive approach**: Numerical gradients require $2n$ forward passes for $n$ parameters

    **Solution**: Backpropagation computes ALL gradients in ONE backward pass!
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate computational cost
    def numerical_gradient_cost(n_params, n_samples):
        forward_passes = 2 * n_params  # Central difference
        return forward_passes

    def backprop_cost(n_params, n_samples):
        return 2  # One forward + one backward

    params = [100, 1000, 10000, 100000, 1000000]

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    numerical_costs = [numerical_gradient_cost(p, 1000) for p in params]
    backprop_costs = [backprop_cost(p, 1000) for p in params]

    x_pos = np.arange(len(params))
    width = 0.35

    ax1.bar(x_pos - width/2, numerical_costs, width, label='Numerical Gradients', color='red', alpha=0.7)
    ax1.bar(x_pos + width/2, backprop_costs, width, label='Backpropagation', color='green', alpha=0.7)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{p:,}' for p in params])
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Forward Passes Required')
    ax1.set_title('Computational Cost: Numerical vs Backpropagation')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    fig1
    return (
        ax1,
        backprop_cost,
        backprop_costs,
        fig1,
        numerical_costs,
        numerical_gradient_cost,
        params,
        width,
        x_pos,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 5.2 Computational Graphs

    Neural network computation can be represented as a **directed acyclic graph (DAG)**:
    - Nodes: Operations or variables
    - Edges: Data flow

    **Forward pass**: Compute outputs (left → right)
    **Backward pass**: Compute gradients (right → left)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5.3 The Chain Rule in Action

    For $L = f(g(h(x)))$:
    $$\frac{\partial L}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

    Each node:
    1. Computes **local gradient** (derivative of its operation)
    2. Multiplies by **upstream gradient** (from output)
    """)
    return


@app.cell
def _(np, plt):
    # Visualize backpropagation on a simple network
    # y = sigmoid(w2 * relu(w1 * x + b1) + b2)

    class ComputeNode:
        def __init__(self, name):
            self.name = name
            self.output = None
            self.grad = None
        
        def __repr__(self):
            return f"{self.name}: out={self.output:.4f}, grad={self.grad:.4f}" if self.output else self.name

    def forward_backward_trace():
        # Input and weights
        x = 2.0
        w1, b1 = 0.5, 0.1
        w2, b2 = 0.8, -0.2
        target = 0.7
        
        # Forward pass (with tracing)
        nodes = {}
        
        # z1 = w1 * x + b1
        z1 = w1 * x + b1
        nodes['z1'] = z1
        
        # h1 = relu(z1)
        h1 = max(0, z1)
        nodes['h1'] = h1
        
        # z2 = w2 * h1 + b2
        z2 = w2 * h1 + b2
        nodes['z2'] = z2
        
        # y = sigmoid(z2)
        y = 1 / (1 + np.exp(-z2))
        nodes['y'] = y
        
        # Loss = (y - target)^2
        L = (y - target) ** 2
        nodes['L'] = L
        
        # Backward pass
        grads = {}
        
        # dL/dy
        dL_dy = 2 * (y - target)
        grads['y'] = dL_dy
        
        # dL/dz2 = dL/dy * dy/dz2 = dL/dy * y*(1-y)
        dy_dz2 = y * (1 - y)
        dL_dz2 = dL_dy * dy_dz2
        grads['z2'] = dL_dz2
        
        # dL/dw2 = dL/dz2 * dz2/dw2 = dL/dz2 * h1
        dL_dw2 = dL_dz2 * h1
        grads['w2'] = dL_dw2
        
        # dL/db2 = dL/dz2
        grads['b2'] = dL_dz2
        
        # dL/dh1 = dL/dz2 * dz2/dh1 = dL/dz2 * w2
        dL_dh1 = dL_dz2 * w2
        grads['h1'] = dL_dh1
        
        # dL/dz1 = dL/dh1 * dh1/dz1 (relu derivative)
        dh1_dz1 = 1 if z1 > 0 else 0
        dL_dz1 = dL_dh1 * dh1_dz1
        grads['z1'] = dL_dz1
        
        # dL/dw1 = dL/dz1 * dz1/dw1 = dL/dz1 * x
        dL_dw1 = dL_dz1 * x
        grads['w1'] = dL_dw1
        
        # dL/db1 = dL/dz1
        grads['b1'] = dL_dz1
        
        return nodes, grads

    nodes, grads = forward_backward_trace()

    # Create visualization
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    # Node positions
    positions = {
        'x': (0, 0.5),
        'z1': (1, 0.5),
        'h1': (2, 0.5),
        'z2': (3, 0.5),
        'y': (4, 0.5),
        'L': (5, 0.5)
    }

    # Draw nodes
    for name, (px, py) in positions.items():
        if name in nodes:
            val = nodes[name]
            grad = grads.get(name, 0)
            color = 'lightblue' if name == 'x' else ('lightgreen' if name == 'L' else 'lightyellow')
            
            circle = plt.Circle((px, py), 0.15, fill=True, color=color, edgecolor='black', linewidth=2)
            ax2.add_patch(circle)
            ax2.text(px, py + 0.02, name, ha='center', va='center', fontsize=12, fontweight='bold')
            ax2.text(px, py - 0.25, f'val={val:.3f}', ha='center', fontsize=9)
            if name in grads:
                ax2.text(px, py + 0.25, f'grad={grad:.4f}', ha='center', fontsize=9, color='red')

    # Draw input x
    circle = plt.Circle((0, 0.5), 0.15, fill=True, color='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    ax2.text(0, 0.52, 'x=2', ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw edges with operations
    edges = [
        ((0, 0.5), (1, 0.5), 'w1·x+b1'),
        ((1, 0.5), (2, 0.5), 'ReLU'),
        ((2, 0.5), (3, 0.5), 'w2·h+b2'),
        ((3, 0.5), (4, 0.5), 'σ'),
        ((4, 0.5), (5, 0.5), 'MSE'),
    ]

    for (x1, y1), (x2, y2), op in edges:
        ax2.annotate('', xy=(x2 - 0.15, y2), xytext=(x1 + 0.15, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax2.text((x1 + x2) / 2, y1 + 0.12, op, ha='center', fontsize=10, style='italic')

    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Backpropagation: Forward Values (black) and Backward Gradients (red)', fontsize=14)

    fig2
    return (
        ComputeNode,
        ax2,
        circle,
        edges,
        fig2,
        forward_backward_trace,
        grads,
        nodes,
        op,
        positions,
        x1,
        x2,
        y1,
        y2,
    )


@app.cell
def _(grads, np):
    # Print the gradients
    print("Gradients computed via backpropagation:")
    print("-" * 40)
    for name, grad in grads.items():
        print(f"∂L/∂{name} = {grad:.6f}")
    return grad, name


@app.cell
def _(mo):
    mo.md(r"""
    ## 5.4 Layer-by-Layer Gradients

    ### Linear Layer: $z = Wx + b$
    - $\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial z}$
    - $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$
    - $\frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial z}$

    ### ReLU: $h = \max(0, z)$
    - $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot \mathbb{1}[z > 0]$
    """)
    return


@app.cell
def _(np):
    # Full backprop implementation
    class Layer:
        def forward(self, x):
            raise NotImplementedError
        def backward(self, grad_output):
            raise NotImplementedError

    class Linear(Layer):
        def __init__(self, in_features, out_features):
            # He initialization
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
            self.b = np.zeros(out_features)
            self.grad_W = None
            self.grad_b = None
        
        def forward(self, x):
            self.x = x  # Cache for backward
            return x @ self.W + self.b
        
        def backward(self, grad_output):
            batch_size = self.x.shape[0]
            self.grad_W = self.x.T @ grad_output / batch_size
            self.grad_b = np.mean(grad_output, axis=0)
            return grad_output @ self.W.T

    class ReLU(Layer):
        def forward(self, x):
            self.mask = (x > 0)
            return x * self.mask
        
        def backward(self, grad_output):
            return grad_output * self.mask

    class Sigmoid(Layer):
        def forward(self, x):
            self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return self.output
        
        def backward(self, grad_output):
            return grad_output * self.output * (1 - self.output)

    class MSELoss:
        def forward(self, pred, target):
            self.pred = pred
            self.target = target
            return np.mean((pred - target) ** 2)
        
        def backward(self):
            return 2 * (self.pred - self.target) / self.pred.shape[0]

    print("Layer classes defined: Linear, ReLU, Sigmoid, MSELoss")
    return Layer, Linear, MSELoss, ReLU, Sigmoid


@app.cell
def _(Linear, MSELoss, ReLU, np, plt):
    # Test the implementation
    class SimpleNet:
        def __init__(self, layer_sizes):
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes) - 2:  # No activation after last layer
                    self.layers.append(ReLU())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
        
        def backward(self, grad):
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
        
        def update(self, lr):
            for layer in self.layers:
                if isinstance(layer, Linear):
                    layer.W -= lr * layer.grad_W
                    layer.b -= lr * layer.grad_b

    # Generate data
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0]**2 + X_train[:, 1]**2 < 1).astype(float).reshape(-1, 1)

    # Train
    model = SimpleNet([2, 16, 8, 1])
    loss_fn = MSELoss()

    losses = []
    for epoch in range(500):
        # Forward
        pred = model.forward(X_train)
        loss = loss_fn.forward(pred, y_train)
        losses.append(loss)
        
        # Backward
        grad = loss_fn.backward()
        model.backward(grad)
        
        # Update
        model.update(lr=0.5)

    # Plot training
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    axes3[0].plot(losses, 'b-', linewidth=1)
    axes3[0].set_xlabel('Epoch')
    axes3[0].set_ylabel('Loss')
    axes3[0].set_title('Training Loss (Backpropagation from Scratch)')
    axes3[0].grid(True, alpha=0.3)

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)

    axes3[1].contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.7)
    axes3[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    axes3[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='RdYlBu', 
                     edgecolors='black', s=30, alpha=0.7)
    axes3[1].set_xlabel('x₁')
    axes3[1].set_ylabel('x₂')
    axes3[1].set_title('Learned Decision Boundary')

    plt.tight_layout()
    fig3
    return (
        SimpleNet,
        X_train,
        Z,
        axes3,
        epoch,
        fig3,
        grad,
        grid,
        loss,
        loss_fn,
        losses,
        model,
        pred,
        xx,
        y_train,
        yy,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 5.5 Gradient Checking

    **Always verify analytical gradients numerically!**

    $$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

    Relative error should be $< 10^{-5}$
    """)
    return


@app.cell
def _(Linear, MSELoss, ReLU, np, plt):
    # Gradient checking
    def gradient_check(layer, x, upstream_grad, eps=1e-5):
        """Check if analytical gradient matches numerical gradient."""
        
        # Analytical gradient
        output = layer.forward(x)
        layer.backward(upstream_grad)
        
        results = []
        
        if hasattr(layer, 'W'):
            # Check W gradient
            analytical_W = layer.grad_W.copy()
            numerical_W = np.zeros_like(layer.W)
            
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    original = layer.W[i, j]
                    
                    layer.W[i, j] = original + eps
                    out_plus = layer.forward(x)
                    loss_plus = np.sum(out_plus * upstream_grad)
                    
                    layer.W[i, j] = original - eps
                    out_minus = layer.forward(x)
                    loss_minus = np.sum(out_minus * upstream_grad)
                    
                    layer.W[i, j] = original
                    numerical_W[i, j] = (loss_plus - loss_minus) / (2 * eps)
            
            rel_error_W = np.linalg.norm(analytical_W - numerical_W) / (
                np.linalg.norm(analytical_W) + np.linalg.norm(numerical_W) + 1e-8)
            results.append(('W', rel_error_W))
        
        return results

    # Test gradient checking
    np.random.seed(42)
    test_layer = Linear(4, 3)
    test_x = np.random.randn(5, 4)
    test_upstream = np.random.randn(5, 3)

    check_results = gradient_check(test_layer, test_x, test_upstream)

    print("Gradient Checking Results:")
    print("-" * 40)
    for param_name, rel_error in check_results:
        status = "✓ PASS" if rel_error < 1e-5 else "✗ FAIL"
        print(f"{param_name}: Relative Error = {rel_error:.2e} {status}")
    return (
        check_results,
        gradient_check,
        param_name,
        rel_error,
        status,
        test_layer,
        test_upstream,
        test_x,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 5.6 Vanishing and Exploding Gradients

    In deep networks, gradients can become very small (vanishing) or very large (exploding):

    $$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \prod_{l=1}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}}$$

    **Solutions**: ReLU, proper initialization, skip connections, normalization
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate vanishing gradients with sigmoid vs ReLU
    def simulate_gradient_flow(n_layers, activation='sigmoid'):
        np.random.seed(42)
        
        gradients = [1.0]  # Start with upstream gradient of 1
        
        for l in range(n_layers):
            # Random weights
            W = np.random.randn(100, 100) * 0.1
            
            # Random activations (for derivative computation)
            if activation == 'sigmoid':
                h = 1 / (1 + np.exp(-np.random.randn(100)))
                derivative = np.mean(h * (1 - h))  # sigmoid derivative
            else:  # relu
                z = np.random.randn(100)
                derivative = np.mean(z > 0)  # relu derivative
            
            # Gradient magnitude after this layer
            grad_magnitude = gradients[-1] * np.abs(W).mean() * derivative
            gradients.append(grad_magnitude)
        
        return gradients

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    n_layers_test = 20

    # Sigmoid
    grads_sigmoid = simulate_gradient_flow(n_layers_test, 'sigmoid')
    axes4[0].semilogy(grads_sigmoid, 'r.-', linewidth=2, markersize=8)
    axes4[0].set_xlabel('Layer (from output)')
    axes4[0].set_ylabel('Gradient Magnitude (log scale)')
    axes4[0].set_title('Sigmoid: Vanishing Gradients')
    axes4[0].grid(True, alpha=0.3)
    axes4[0].axhline(1e-10, color='gray', linestyle='--', alpha=0.5, label='Numerical precision limit')
    axes4[0].legend()

    # ReLU
    grads_relu = simulate_gradient_flow(n_layers_test, 'relu')
    axes4[1].semilogy(grads_relu, 'g.-', linewidth=2, markersize=8)
    axes4[1].set_xlabel('Layer (from output)')
    axes4[1].set_ylabel('Gradient Magnitude (log scale)')
    axes4[1].set_title('ReLU: Better Gradient Flow')
    axes4[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig4
    return (
        axes4,
        fig4,
        grads_relu,
        grads_sigmoid,
        n_layers_test,
        simulate_gradient_flow,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | Key Point |
    |---------|-----------|
    | **Backpropagation** | Compute all gradients in one backward pass |
    | **Chain Rule** | Multiply local gradient by upstream gradient |
    | **Gradient Checking** | Verify analytical vs numerical gradients |
    | **Vanishing Gradients** | Use ReLU, proper init, skip connections |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 6.
    - **Classic**: Rumelhart, Hinton & Williams (1986). "Learning representations by back-propagating errors."

    ## Connection to ML Refined Curriculum

    Backpropagation enables training for:
    - All gradient descent methods (Weeks 2-3)
    - Any supervised learning model (Weeks 4-8)
    """)
    return


if __name__ == "__main__":
    app.run()

