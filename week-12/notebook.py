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
    # Week 12: Kernel Methods & Neural Networks

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Watt, Borhani, & Katsaggelos (2020). *Machine Learning Refined* (2nd ed.), **Chapters 12-13**

    ---

    ## Learning Objectives

    - Understand the kernel trick
    - Apply kernel methods for nonlinear classification
    - Understand neural network architecture
    - Implement forward propagation and backpropagation
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    return SVC, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Kernel Methods (Chapter 12)

    ### The Key Insight (Section 12.2-12.3)

    Many algorithms depend on data only through **inner products** $x_i^T x_j$.

    The **kernel trick**: Replace inner product with kernel function:
    $$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

    Without explicitly computing $\phi(x)$!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Common Kernels (Section 12.4)

    | Kernel | Formula | Use Case |
    |--------|---------|----------|
    | **Linear** | $x^T x'$ | Linear relationships |
    | **Polynomial** | $(1 + x^T x')^d$ | Polynomial features |
    | **RBF (Gaussian)** | $\exp(-\gamma \|x - x'\|^2)$ | Most common, general |
    | **Sigmoid** | $\tanh(\alpha x^T x' + c)$ | Neural network-like |
    """)
    return


@app.cell
def _(SVC, np, plt):
    # Kernel SVM comparison
    np.random.seed(42)
    n = 200
    
    # Generate non-linearly separable data (circles)
    theta = np.random.uniform(0, 2*np.pi, n)
    r_inner = 1 + 0.3 * np.random.randn(n//2)
    r_outer = 3 + 0.3 * np.random.randn(n//2)
    
    X_inner = np.column_stack([r_inner * np.cos(theta[:n//2]), r_inner * np.sin(theta[:n//2])])
    X_outer = np.column_stack([r_outer * np.cos(theta[n//2:]), r_outer * np.sin(theta[n//2:])])
    X = np.vstack([X_inner, X_outer])
    y = np.array([0]*(n//2) + [1]*(n//2))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kernels = ['linear', 'poly', 'rbf']
    
    for ax, kernel in zip(axes, kernels):
        svm = SVC(kernel=kernel, gamma='auto')
        svm.fit(X, y)
        
        # Decision boundary
        xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=20, alpha=0.7)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=20, alpha=0.7)
        ax.set_title(f'{kernel.upper()} Kernel')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    fig.suptitle('Kernel SVM Comparison (ML Refined, Chapter 12)', fontsize=14)
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Neural Networks (Chapter 13)

    ### Fully Connected Neural Networks (Section 13.2)

    A neural network composes linear transformations with nonlinear activations:

    $$h^{(1)} = \sigma(W^{(1)} x + b^{(1)})$$
    $$h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)})$$
    $$\vdots$$
    $$f(x) = W^{(L)} h^{(L-1)} + b^{(L)}$$

    Where $\sigma$ is a nonlinear activation function.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Activation Functions (Section 13.3)

    | Function | Formula | Properties |
    |----------|---------|------------|
    | **Sigmoid** | $\frac{1}{1+e^{-z}}$ | Output in (0,1), vanishing gradient |
    | **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Output in (-1,1), zero-centered |
    | **ReLU** | $\max(0, z)$ | Simple, no vanishing gradient |
    | **Leaky ReLU** | $\max(\alpha z, z)$ | No dying neurons |
    """)
    return


@app.cell
def _(np, plt):
    # Activation functions
    z = np.linspace(-5, 5, 100)
    
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)
    leaky_relu = np.where(z > 0, z, 0.1 * z)
    
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    funcs = [(sigmoid, 'Sigmoid'), (tanh, 'Tanh'), 
             (relu, 'ReLU'), (leaky_relu, 'Leaky ReLU')]
    
    for ax, (func, name) in zip(axes.flat, funcs):
        ax.plot(z, func, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('z')
        ax.set_ylabel(f'{name}(z)')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    fig2.suptitle('Activation Functions (ML Refined, Section 13.3)', fontsize=14)
    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Backpropagation Algorithm (Section 13.4)

    ### Forward Pass

    Compute all layer outputs from input to output.

    ### Backward Pass (Chain Rule)

    $$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}$$

    Propagate gradients from output back to input.

    ### Key Insight

    Chain rule enables efficient gradient computation in $O(n)$ operations, where $n$ is the number of parameters.
    """)
    return


@app.cell
def _(np, plt):
    # Simple neural network visualization
    def draw_neural_network(ax, layer_sizes):
        v_spacing = 1.0
        h_spacing = 2.0
        
        # Draw nodes
        for i, size in enumerate(layer_sizes):
            x = i * h_spacing
            y_offset = (max(layer_sizes) - size) / 2
            
            for j in range(size):
                y = j * v_spacing + y_offset
                circle = plt.Circle((x, y), 0.3, color='steelblue', ec='black')
                ax.add_patch(circle)
                
                # Draw connections to next layer
                if i < len(layer_sizes) - 1:
                    next_size = layer_sizes[i + 1]
                    next_y_offset = (max(layer_sizes) - next_size) / 2
                    for k in range(next_size):
                        next_y = k * v_spacing + next_y_offset
                        ax.plot([x + 0.3, (i + 1) * h_spacing - 0.3], 
                               [y, next_y], 'gray', linewidth=0.5, alpha=0.5)
        
        ax.set_xlim(-1, (len(layer_sizes) - 1) * h_spacing + 1)
        ax.set_ylim(-1, max(layer_sizes) * v_spacing)
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    draw_neural_network(ax3, [4, 6, 4, 2])
    ax3.set_title('Fully Connected Neural Network: [4, 6, 4, 2]', fontsize=14)
    
    # Add labels
    labels = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    for i, label in enumerate(labels):
        ax3.text(i * 2, -1.5, label, ha='center', fontsize=10)
    
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Batch Normalization (Section 13.6)

    Normalize activations within each layer:

    $$\hat{h} = \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
    $$\tilde{h} = \gamma \hat{h} + \beta$$

    ### Benefits

    - Faster training
    - Higher learning rates
    - Some regularization effect
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Key Idea | Use Case |
    |--------|----------|----------|
    | **Kernel Methods** | Implicit feature mapping | Small-medium data |
    | **Neural Networks** | Learned feature hierarchy | Large data, complex patterns |
    | **Backpropagation** | Efficient gradient computation | Training NNs |

    ---

    ## References

    - **Primary**: Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), Chapters 12-13.
    - **Supplementary**: Goodfellow, I. et al. (2016). *Deep Learning*, Chapters 5-6.

    ## Next Week

    **Tree-Based Learners & Advanced Topics** (Chapter 14): Decision trees and ensemble methods.
    """)
    return


if __name__ == "__main__":
    app.run()
