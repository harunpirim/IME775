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
    # Week 7: Deep Architectures - Modern Building Blocks

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 8

    ---

    ## Learning Objectives

    - Understand the degradation problem in deep networks
    - Master residual connections
    - Learn dense connections and SE blocks
    - Understand modern efficient architectures
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
    ## 7.1 The Degradation Problem

    **Paradox**: Deeper networks can perform **worse** than shallow ones, even on training data!

    This isn't overfitting - it's an optimization problem.
    """)
    return


@app.cell
def _(np, plt):
    # Simulate degradation problem
    np.random.seed(42)

    # Simulated training curves
    epochs = 100
    
    # Shallow network (20 layers) - good convergence
    shallow_train = 0.5 * np.exp(-epochs * np.linspace(0, 1, epochs) * 3) + 0.04 + np.random.randn(epochs) * 0.005
    shallow_val = 0.5 * np.exp(-epochs * np.linspace(0, 1, epochs) * 2.5) + 0.06 + np.random.randn(epochs) * 0.008
    
    # Deep network (56 layers) without skip connections - degradation
    deep_train = 0.5 * np.exp(-epochs * np.linspace(0, 1, epochs) * 1.5) + 0.08 + np.random.randn(epochs) * 0.005
    deep_val = 0.5 * np.exp(-epochs * np.linspace(0, 1, epochs) * 1.2) + 0.12 + np.random.randn(epochs) * 0.01

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Training error
    axes1[0].plot(shallow_train, 'b-', linewidth=2, label='20-layer network')
    axes1[0].plot(deep_train, 'r-', linewidth=2, label='56-layer network')
    axes1[0].set_xlabel('Epoch')
    axes1[0].set_ylabel('Training Error')
    axes1[0].set_title('Training Error: Deeper ≠ Better!')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # Validation error
    axes1[1].plot(shallow_val, 'b-', linewidth=2, label='20-layer network')
    axes1[1].plot(deep_val, 'r-', linewidth=2, label='56-layer network')
    axes1[1].set_xlabel('Epoch')
    axes1[1].set_ylabel('Validation Error')
    axes1[1].set_title('Validation Error: Degradation Problem')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig1
    return (
        axes1,
        deep_train,
        deep_val,
        epochs,
        fig1,
        shallow_train,
        shallow_val,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7.2 Residual Networks (ResNet)

    **Key Insight**: Instead of learning $H(x)$, learn the residual $F(x) = H(x) - x$

    $$H(x) = F(x) + x$$

    The skip connection adds $x$ directly to the output!
    """)
    return


@app.cell
def _(np, plt):
    # Visualize residual block
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Plain network block
    ax1 = axes2[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Draw plain block
    ax1.add_patch(plt.Rectangle((3, 7), 4, 1.5, fill=True, facecolor='lightblue', edgecolor='black'))
    ax1.text(5, 7.75, 'Conv-BN-ReLU', ha='center', va='center', fontsize=10)
    
    ax1.add_patch(plt.Rectangle((3, 4), 4, 1.5, fill=True, facecolor='lightblue', edgecolor='black'))
    ax1.text(5, 4.75, 'Conv-BN-ReLU', ha='center', va='center', fontsize=10)
    
    # Arrows
    ax1.annotate('', xy=(5, 7), xytext=(5, 5.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.annotate('', xy=(5, 4), xytext=(5, 2.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.annotate('', xy=(5, 9.5), xytext=(5, 8.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.text(5, 9.7, 'x', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 2.3, 'H(x)', ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_title('Plain Block: Learn H(x) directly', fontsize=12)
    ax1.axis('off')

    # Residual block
    ax2 = axes2[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Draw residual block
    ax2.add_patch(plt.Rectangle((3, 7), 4, 1.5, fill=True, facecolor='lightgreen', edgecolor='black'))
    ax2.text(5, 7.75, 'Conv-BN-ReLU', ha='center', va='center', fontsize=10)
    
    ax2.add_patch(plt.Rectangle((3, 4), 4, 1.5, fill=True, facecolor='lightgreen', edgecolor='black'))
    ax2.text(5, 4.75, 'Conv-BN', ha='center', va='center', fontsize=10)
    
    # Skip connection
    ax2.plot([1.5, 1.5], [9, 2.5], 'r-', linewidth=2)
    ax2.annotate('', xy=(5, 2.5), xytext=(1.5, 2.5), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Addition
    ax2.add_patch(plt.Circle((5, 2.5), 0.3, fill=True, facecolor='yellow', edgecolor='black'))
    ax2.text(5, 2.5, '+', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Arrows
    ax2.annotate('', xy=(5, 7), xytext=(5, 5.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.annotate('', xy=(5, 4), xytext=(5, 2.8), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.annotate('', xy=(5, 9.5), xytext=(5, 8.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.annotate('', xy=(5, 2.2), xytext=(5, 1), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.plot([1.5, 5], [9, 9], 'black', linewidth=1)
    ax2.plot([1.5, 1.5], [9, 9], 'black', linewidth=1)
    
    ax2.text(5, 9.7, 'x', ha='center', fontsize=12, fontweight='bold')
    ax2.text(5, 0.7, 'H(x) = F(x) + x', ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.8, 5.5, 'Skip\nConnection', ha='center', fontsize=10, color='red')
    
    ax2.set_title('Residual Block: Learn F(x) = H(x) - x', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    fig2
    return ax1, ax2, axes2, fig2


@app.cell
def _(mo):
    mo.md(r"""
    ## 7.3 Why Residual Connections Work

    **Gradient Flow Analysis**:

    Without skip: $\frac{\partial L}{\partial x_l} = \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i}$ → Can vanish!

    With skip: $\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L}(1 + \text{other terms})$ → Always has the "1"!
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate gradient flow
    def simulate_gradient_flow_resnet(n_layers, use_skip=True):
        np.random.seed(42)
        gradients = [1.0]
        
        for _ in range(n_layers):
            # Residual function gradient (can be small)
            f_grad = np.random.uniform(0.1, 0.5)
            
            if use_skip:
                # With skip: gradient = 1 + f_grad
                total_grad = gradients[-1] * (1.0 + f_grad * 0.1)
            else:
                # Without skip: gradient = f_grad only
                total_grad = gradients[-1] * f_grad
            
            gradients.append(total_grad)
        
        return gradients

    n_layers_demo = 50

    grad_no_skip = simulate_gradient_flow_resnet(n_layers_demo, use_skip=False)
    grad_with_skip = simulate_gradient_flow_resnet(n_layers_demo, use_skip=True)

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.semilogy(grad_no_skip, 'r-', linewidth=2, label='Without Skip Connections')
    ax3.semilogy(grad_with_skip, 'g-', linewidth=2, label='With Skip Connections (ResNet)')
    ax3.axhline(1e-10, color='gray', linestyle='--', alpha=0.5, label='Numerical precision')

    ax3.set_xlabel('Layer (from output to input)')
    ax3.set_ylabel('Gradient Magnitude (log scale)')
    ax3.set_title('Gradient Flow: Skip Connections Prevent Vanishing')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3
    return (
        ax3,
        fig3,
        grad_no_skip,
        grad_with_skip,
        n_layers_demo,
        simulate_gradient_flow_resnet,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7.4 DenseNet: Dense Connections

    **Key Idea**: Connect every layer to every subsequent layer

    $$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

    Concatenate all previous features!
    """)
    return


@app.cell
def _(np, plt):
    # Visualize connectivity patterns
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    def draw_connectivity(ax, connections, title):
        n_layers = len(connections)
        
        # Draw layers
        for i in range(n_layers):
            ax.add_patch(plt.Circle((i, 0), 0.15, fill=True, color='lightblue', edgecolor='black'))
            ax.text(i, 0, str(i), ha='center', va='center', fontsize=10)
        
        # Draw connections
        for i, conns in enumerate(connections):
            for j in conns:
                if j < i:
                    # Draw curved arrow
                    y_offset = 0.3 + 0.15 * (i - j)
                    ax.annotate('', xy=(i-0.15, 0.1), xytext=(j+0.15, 0.1),
                               arrowprops=dict(arrowstyle='->', color='red', 
                                             connectionstyle=f'arc3,rad={0.3}', lw=1))
        
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Plain network
    plain = [[], [0], [1], [2], [3]]
    draw_connectivity(axes4[0], plain, 'Plain Network\n(Sequential)')

    # ResNet
    resnet = [[], [0], [0, 1], [1, 2], [2, 3]]  
    axes4[1].text(0.5, 1.3, 'ResNet: Skip every 2 layers', ha='center', transform=axes4[1].transAxes)
    
    # Draw ResNet manually
    for i in range(5):
        axes4[1].add_patch(plt.Circle((i, 0), 0.15, fill=True, color='lightgreen', edgecolor='black'))
        axes4[1].text(i, 0, str(i), ha='center', va='center', fontsize=10)
    
    # Sequential connections
    for i in range(4):
        axes4[1].annotate('', xy=(i+0.85, 0), xytext=(i+0.15, 0),
                         arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    # Skip connections (every 2)
    for i in range(0, 4, 2):
        axes4[1].annotate('', xy=(i+1.85, 0.1), xytext=(i+0.15, 0.1),
                         arrowprops=dict(arrowstyle='->', color='red', 
                                       connectionstyle='arc3,rad=0.5', lw=1.5))
    
    axes4[1].set_xlim(-0.5, 4.5)
    axes4[1].set_ylim(-0.5, 1.5)
    axes4[1].set_title('ResNet\n(Skip connections)')
    axes4[1].axis('off')

    # DenseNet
    axes4[2].text(0.5, 1.3, 'DenseNet: All-to-all connections', ha='center', transform=axes4[2].transAxes)
    
    for i in range(5):
        axes4[2].add_patch(plt.Circle((i, 0), 0.15, fill=True, color='lightyellow', edgecolor='black'))
        axes4[2].text(i, 0, str(i), ha='center', va='center', fontsize=10)
    
    # All connections
    for i in range(5):
        for j in range(i):
            y_rad = 0.3 + 0.1 * (i - j)
            axes4[2].annotate('', xy=(i-0.15, 0.1), xytext=(j+0.15, 0.1),
                             arrowprops=dict(arrowstyle='->', color='orange', 
                                           connectionstyle=f'arc3,rad={y_rad}', lw=0.8, alpha=0.7))
    
    axes4[2].set_xlim(-0.5, 4.5)
    axes4[2].set_ylim(-0.5, 1.5)
    axes4[2].set_title('DenseNet\n(Dense connections)')
    axes4[2].axis('off')

    plt.tight_layout()
    fig4
    return axes4, draw_connectivity, fig4, i, j, plain, resnet, y_rad


@app.cell
def _(mo):
    mo.md(r"""
    ## 7.5 Squeeze-and-Excitation (SE) Blocks

    **Channel Attention**: Not all channels are equally important.

    1. **Squeeze**: Global average pooling
    2. **Excitation**: Learn channel weights
    3. **Scale**: Reweight channels
    """)
    return


@app.cell
def _(np, plt):
    # Visualize SE block operation
    np.random.seed(42)

    # Simulate feature map
    H, W, C = 4, 4, 8
    feature_map = np.random.randn(H, W, C)

    # Squeeze: Global Average Pooling
    squeezed = feature_map.mean(axis=(0, 1))  # Shape: (C,)

    # Excitation: Simple simulation (normally learned)
    # Two FC layers with reduction ratio r=4
    r = 4
    W1_se = np.random.randn(C, C // r) * 0.5
    W2_se = np.random.randn(C // r, C) * 0.5

    excitation = np.tanh(squeezed @ W1_se)
    channel_weights = 1 / (1 + np.exp(-excitation @ W2_se))  # Sigmoid

    # Scale
    scaled_feature_map = feature_map * channel_weights

    # Visualize
    fig5, axes5 = plt.subplots(1, 4, figsize=(16, 3))

    # Original channel importance (average activation)
    orig_importance = np.abs(feature_map).mean(axis=(0, 1))
    axes5[0].bar(range(C), orig_importance, color='blue', alpha=0.7)
    axes5[0].set_xlabel('Channel')
    axes5[0].set_ylabel('Avg Activation')
    axes5[0].set_title('Original Channel Activations')

    # Squeezed representation
    axes5[1].bar(range(C), squeezed, color='green', alpha=0.7)
    axes5[1].set_xlabel('Channel')
    axes5[1].set_ylabel('Value')
    axes5[1].set_title('After Squeeze (GAP)')

    # Learned weights
    axes5[2].bar(range(C), channel_weights, color='orange', alpha=0.7)
    axes5[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes5[2].set_xlabel('Channel')
    axes5[2].set_ylabel('Weight')
    axes5[2].set_title('Excitation Weights (Learned)')

    # Scaled importance
    scaled_importance = np.abs(scaled_feature_map).mean(axis=(0, 1))
    axes5[3].bar(range(C), scaled_importance, color='red', alpha=0.7)
    axes5[3].set_xlabel('Channel')
    axes5[3].set_ylabel('Avg Activation')
    axes5[3].set_title('After SE (Reweighted)')

    plt.tight_layout()
    fig5
    return (
        C,
        H,
        W,
        W1_se,
        W2_se,
        axes5,
        channel_weights,
        excitation,
        feature_map,
        fig5,
        orig_importance,
        r,
        scaled_feature_map,
        scaled_importance,
        squeezed,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7.6 Efficient Architectures

    **Depthwise Separable Convolution** (MobileNet):
    - Standard: $K^2 \times C_{in} \times C_{out}$ parameters
    - Separable: $K^2 \times C_{in} + C_{in} \times C_{out}$ parameters
    - Reduction: ~$\frac{1}{K^2}$
    """)
    return


@app.cell
def _(np, plt):
    # Compare parameter counts
    def conv_params(k, c_in, c_out):
        return k * k * c_in * c_out

    def depthwise_separable_params(k, c_in, c_out):
        depthwise = k * k * c_in  # One filter per input channel
        pointwise = c_in * c_out  # 1x1 convolution
        return depthwise + pointwise

    # Compare for different configurations
    configs = [
        (3, 64, 64),
        (3, 128, 128),
        (3, 256, 256),
        (3, 512, 512),
        (5, 64, 64),
    ]

    fig6, ax6 = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(configs))
    width = 0.35

    standard = [conv_params(k, c_in, c_out) for k, c_in, c_out in configs]
    separable = [depthwise_separable_params(k, c_in, c_out) for k, c_in, c_out in configs]

    ax6.bar(x_pos - width/2, standard, width, label='Standard Conv', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, separable, width, label='Depthwise Separable', color='green', alpha=0.7)

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'{k}×{k}, {c_in}→{c_out}' for k, c_in, c_out in configs], rotation=15)
    ax6.set_ylabel('Parameters')
    ax6.set_title('Parameter Reduction: Depthwise Separable Convolutions')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add reduction ratios
    for i, (s, sep) in enumerate(zip(standard, separable)):
        ratio = sep / s
        ax6.text(i, max(s, sep) + 5000, f'{ratio:.2%}', ha='center', fontsize=9)

    fig6
    return (
        ax6,
        configs,
        conv_params,
        depthwise_separable_params,
        fig6,
        i,
        ratio,
        s,
        sep,
        separable,
        standard,
        width,
        x_pos,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Architecture | Key Innovation | Use Case |
    |--------------|----------------|----------|
    | **ResNet** | Skip connections | Very deep networks |
    | **DenseNet** | Dense connections | Feature reuse |
    | **SE-Net** | Channel attention | Any backbone |
    | **MobileNet** | Depthwise separable | Mobile/edge devices |
    | **EfficientNet** | Compound scaling | Best accuracy/compute |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 8.
    - **ResNet**: He et al. (2016). "Deep Residual Learning."
    - **DenseNet**: Huang et al. (2017). "Densely Connected Networks."
    - **SE-Net**: Hu et al. (2018). "Squeeze-and-Excitation Networks."

    ## Connection to ML Refined Curriculum

    Modern architectures extend concepts from:
    - Week 7: Multi-class classification
    - Week 8: Feature learning (PCA → learned features)
    """)
    return


if __name__ == "__main__":
    app.run()

