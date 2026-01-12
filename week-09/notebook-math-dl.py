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
    # Week 9: Convolutional Neural Networks (CNNs)

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 9

    ---

    ## Learning Objectives

    - Understand the convolution operation mathematically
    - Master CNN building blocks: convolution, pooling, padding
    - Learn classic architectures
    - Visualize what CNNs learn
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    return np, plt, signal


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.1 Why Convolutions?

    **Problems with Fully Connected layers for images:**
    - Too many parameters (224×224×3 image → 150M params for first layer!)
    - No spatial structure
    - No translation invariance

    **Convolution advantages:**
    - Parameter sharing (same filter everywhere)
    - Local connectivity
    - Translation equivariance
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate convolution operation
    def conv2d_manual(X, W, stride=1, padding=0):
        """Simple 2D convolution implementation."""
        k = W.shape[0]
        
        if padding > 0:
            X = np.pad(X, padding, mode='constant')
        
        H, W_in = X.shape
        H_out = (H - k) // stride + 1
        W_out = (W_in - k) // stride + 1
        
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                patch = X[i*stride:i*stride+k, j*stride:j*stride+k]
                output[i, j] = np.sum(patch * W)
        
        return output

    # Create a simple image
    np.random.seed(42)
    image = np.zeros((8, 8))
    image[2:6, 2:6] = 1  # Square in the middle

    # Different filters
    filters = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Edge (Horizontal)': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Edge (Vertical)': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }

    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))

    # Original image
    axes1[0, 0].imshow(image, cmap='gray')
    axes1[0, 0].set_title('Original Image (8×8)')
    axes1[0, 0].axis('off')

    # Filter visualizations
    for idx, (name, filt) in enumerate(list(filters.items())[:2]):
        ax = axes1[0, idx + 1]
        ax.imshow(filt, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_title(f'Filter: {name}')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{filt[i,j]:.0f}', ha='center', va='center', fontsize=12)
        ax.axis('off')

    # Convolution results
    for idx, (name, filt) in enumerate(filters.items()):
        row = 1 if idx >= 2 else 0
        col = idx if idx < 2 else idx - 2
        if idx >= 2:
            ax = axes1[1, col]
        else:
            continue
            
        result = conv2d_manual(image, filt, padding=1)
        ax.imshow(result, cmap='gray')
        ax.set_title(f'After {name} Filter')
        ax.axis('off')

    # Add results for first two filters
    for idx, (name, filt) in enumerate(list(filters.items())[:2]):
        result = conv2d_manual(image, filt, padding=1)
        axes1[1, idx].imshow(result, cmap='gray')
        axes1[1, idx].set_title(f'After {name}')
        axes1[1, idx].axis('off')

    # Show edge detection combined
    h_edge = conv2d_manual(image, filters['Edge (Horizontal)'], padding=1)
    v_edge = conv2d_manual(image, filters['Edge (Vertical)'], padding=1)
    combined = np.sqrt(h_edge**2 + v_edge**2)
    axes1[1, 2].imshow(combined, cmap='gray')
    axes1[1, 2].set_title('Combined Edge Detection')
    axes1[1, 2].axis('off')

    plt.tight_layout()
    fig1
    return (
        axes1,
        combined,
        conv2d_manual,
        fig1,
        filt,
        filters,
        h_edge,
        image,
        name,
        result,
        v_edge,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.2 The Convolution Operation

    For 2D input $X$ and kernel $W$ of size $k \times k$:

    $$(X * W)[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m, j+n] \cdot W[m,n]$$

    **Sliding the filter across the image and computing dot products.**
    """)
    return


@app.cell
def _(conv2d_manual, np, plt):
    # Visualize convolution step by step
    def visualize_conv_step(X, W, step_i, step_j, padding=0):
        k = W.shape[0]
        if padding > 0:
            X_padded = np.pad(X, padding, mode='constant')
        else:
            X_padded = X
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Input with highlight
        ax1 = axes[0]
        ax1.imshow(X_padded, cmap='gray', vmin=0, vmax=1)
        # Highlight the patch
        rect = plt.Rectangle((step_j - 0.5, step_i - 0.5), k, k, 
                             fill=False, edgecolor='red', linewidth=3)
        ax1.add_patch(rect)
        ax1.set_title(f'Input (patch at [{step_i},{step_j}])')
        ax1.axis('off')
        
        # Extracted patch
        ax2 = axes[1]
        patch = X_padded[step_i:step_i+k, step_j:step_j+k]
        ax2.imshow(patch, cmap='gray', vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax2.text(j, i, f'{patch[i,j]:.1f}', ha='center', va='center', 
                        fontsize=12, color='red' if patch[i,j] > 0.5 else 'blue')
        ax2.set_title('Extracted Patch')
        ax2.axis('off')
        
        # Filter
        ax3 = axes[2]
        ax3.imshow(W, cmap='RdBu', vmin=-1, vmax=1)
        for i in range(k):
            for j in range(k):
                ax3.text(j, i, f'{W[i,j]:.1f}', ha='center', va='center', fontsize=12)
        ax3.set_title('Filter (Kernel)')
        ax3.axis('off')
        
        # Element-wise product
        ax4 = axes[3]
        product = patch * W
        ax4.imshow(product, cmap='RdBu', vmin=-1, vmax=1)
        for i in range(k):
            for j in range(k):
                ax4.text(j, i, f'{product[i,j]:.2f}', ha='center', va='center', fontsize=10)
        ax4.set_title(f'Element-wise Product\nSum = {product.sum():.2f}')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig

    # Create example
    X_demo = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)

    W_demo = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=float) / 3

    fig2 = visualize_conv_step(X_demo, W_demo, step_i=1, step_j=0, padding=0)
    fig2
    return W_demo, X_demo, fig2, visualize_conv_step


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.3 Stride and Padding

    **Stride**: Step size when sliding the filter
    $$\text{Output size} = \lfloor \frac{n - k}{\text{stride}} \rfloor + 1$$

    **Padding**: Add zeros around input
    $$\text{Output size} = \lfloor \frac{n + 2p - k}{\text{stride}} \rfloor + 1$$
    """)
    return


@app.cell
def _(conv2d_manual, np, plt):
    # Demonstrate stride and padding effects
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))

    # Input image
    input_img = np.random.rand(8, 8)
    kernel = np.ones((3, 3)) / 9  # Average filter

    configs = [
        ('Stride=1, Pad=0', 1, 0),
        ('Stride=2, Pad=0', 2, 0),
        ('Stride=1, Pad=1 (Same)', 1, 1),
    ]

    # Original
    axes3[0, 0].imshow(input_img, cmap='viridis')
    axes3[0, 0].set_title(f'Input: {input_img.shape[0]}×{input_img.shape[1]}')
    axes3[0, 0].axis('off')

    # Different configurations
    for idx, (name, stride, pad) in enumerate(configs):
        if pad > 0:
            padded = np.pad(input_img, pad, mode='constant')
        else:
            padded = input_img
        
        k = kernel.shape[0]
        H_out = (padded.shape[0] - k) // stride + 1
        W_out = (padded.shape[1] - k) // stride + 1
        
        output = np.zeros((H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                patch = padded[i*stride:i*stride+k, j*stride:j*stride+k]
                output[i, j] = np.sum(patch * kernel)
        
        if idx == 0:
            ax = axes3[0, 1]
        elif idx == 1:
            ax = axes3[0, 2]
        else:
            ax = axes3[1, 0]
        
        ax.imshow(output, cmap='viridis')
        ax.set_title(f'{name}\nOutput: {H_out}×{W_out}')
        ax.axis('off')

    # Show formula
    axes3[1, 1].text(0.5, 0.7, 'Output Size Formula:', ha='center', fontsize=14, fontweight='bold')
    axes3[1, 1].text(0.5, 0.4, r'$\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$', 
                     ha='center', fontsize=20)
    axes3[1, 1].text(0.5, 0.15, 'n=input, k=kernel, p=padding, s=stride', ha='center', fontsize=11)
    axes3[1, 1].axis('off')

    # Example calculation
    axes3[1, 2].text(0.5, 0.8, 'Example: 224×224 input', ha='center', fontsize=12, fontweight='bold')
    axes3[1, 2].text(0.5, 0.6, '7×7 kernel, stride=2, pad=3', ha='center', fontsize=11)
    axes3[1, 2].text(0.5, 0.4, r'$\left\lfloor \frac{224 + 6 - 7}{2} \right\rfloor + 1$', 
                     ha='center', fontsize=16)
    axes3[1, 2].text(0.5, 0.2, '= 112×112 output', ha='center', fontsize=14, fontweight='bold', color='red')
    axes3[1, 2].axis('off')

    plt.tight_layout()
    fig3
    return (
        H_out,
        W_out,
        ax,
        axes3,
        configs,
        fig3,
        idx,
        input_img,
        kernel,
        name,
        output,
        pad,
        padded,
        patch,
        stride,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.4 Pooling Layers

    **Purpose**: Reduce spatial dimensions, add translation invariance

    | Type | Operation | Use |
    |------|-----------|-----|
    | Max | Take maximum | Most common |
    | Average | Take mean | Some architectures |
    | Global Avg | Mean over all spatial | Before classifier |
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate pooling
    def max_pool(X, pool_size=2, stride=2):
        H, W = X.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                patch = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                output[i, j] = patch.max()
        
        return output

    def avg_pool(X, pool_size=2, stride=2):
        H, W = X.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                patch = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                output[i, j] = patch.mean()
        
        return output

    # Create feature map
    np.random.seed(42)
    feature_map = np.random.rand(8, 8)

    fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4))

    # Original
    im0 = axes4[0].imshow(feature_map, cmap='viridis')
    axes4[0].set_title(f'Input: {feature_map.shape[0]}×{feature_map.shape[1]}')
    plt.colorbar(im0, ax=axes4[0], fraction=0.046)
    axes4[0].axis('off')

    # Max pooling
    max_pooled = max_pool(feature_map)
    im1 = axes4[1].imshow(max_pooled, cmap='viridis')
    axes4[1].set_title(f'Max Pool (2×2, s=2)\n{max_pooled.shape[0]}×{max_pooled.shape[1]}')
    plt.colorbar(im1, ax=axes4[1], fraction=0.046)
    axes4[1].axis('off')

    # Average pooling
    avg_pooled = avg_pool(feature_map)
    im2 = axes4[2].imshow(avg_pooled, cmap='viridis')
    axes4[2].set_title(f'Avg Pool (2×2, s=2)\n{avg_pooled.shape[0]}×{avg_pooled.shape[1]}')
    plt.colorbar(im2, ax=axes4[2], fraction=0.046)
    axes4[2].axis('off')

    # Global average pooling
    gap_value = feature_map.mean()
    axes4[3].text(0.5, 0.5, f'{gap_value:.3f}', ha='center', va='center', fontsize=24, fontweight='bold')
    axes4[3].set_title('Global Avg Pool\n(Single value)')
    axes4[3].set_xlim(0, 1)
    axes4[3].set_ylim(0, 1)
    axes4[3].axis('off')

    plt.tight_layout()
    fig4
    return (
        avg_pool,
        avg_pooled,
        axes4,
        feature_map,
        fig4,
        gap_value,
        im0,
        im1,
        im2,
        max_pool,
        max_pooled,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.5 CNN Architecture Evolution

    | Year | Architecture | Key Innovation |
    |------|--------------|----------------|
    | 1998 | LeNet-5 | First successful CNN |
    | 2012 | AlexNet | ReLU, Dropout, GPU |
    | 2014 | VGGNet | Small 3×3 filters |
    | 2015 | ResNet | Skip connections |
    | 2017 | MobileNet | Depthwise separable |
    """)
    return


@app.cell
def _(np, plt):
    # Visualize feature hierarchy
    fig5, axes5 = plt.subplots(1, 4, figsize=(16, 4))

    # Simulate different layer features
    np.random.seed(42)

    # Layer 1: Edges
    edge_filters = np.array([
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],  # Vertical
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],  # Horizontal
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],  # Diagonal
    ])

    for i, filt in enumerate(edge_filters[:3]):
        ax = axes5[0] if i == 0 else None
        
    axes5[0].set_title('Layer 1: Edges', fontsize=12)
    for i, filt in enumerate(edge_filters):
        y_offset = i * 0.35
        for ii in range(3):
            for jj in range(3):
                color = 'red' if filt[ii, jj] > 0 else ('blue' if filt[ii, jj] < 0 else 'gray')
                axes5[0].add_patch(plt.Rectangle((jj/4 + 0.1, 0.55 - y_offset - ii/4), 
                                                  0.2, 0.2, facecolor=color, alpha=0.5))
    axes5[0].set_xlim(0, 1)
    axes5[0].set_ylim(0, 1)
    axes5[0].axis('off')

    # Layer 2-3: Textures/Parts
    axes5[1].text(0.5, 0.7, '🔲 Corners', fontsize=20, ha='center')
    axes5[1].text(0.5, 0.5, '〰️ Curves', fontsize=20, ha='center')
    axes5[1].text(0.5, 0.3, '⬛ Patterns', fontsize=20, ha='center')
    axes5[1].set_title('Layer 2-3: Textures & Parts', fontsize=12)
    axes5[1].axis('off')

    # Layer 4: Parts
    axes5[2].text(0.5, 0.7, '👁️ Eyes', fontsize=20, ha='center')
    axes5[2].text(0.5, 0.5, '👃 Noses', fontsize=20, ha='center')
    axes5[2].text(0.5, 0.3, '🚗 Wheels', fontsize=20, ha='center')
    axes5[2].set_title('Layer 4: Object Parts', fontsize=12)
    axes5[2].axis('off')

    # Layer 5: Objects
    axes5[3].text(0.5, 0.7, '😺 Cats', fontsize=20, ha='center')
    axes5[3].text(0.5, 0.5, '🚙 Cars', fontsize=20, ha='center')
    axes5[3].text(0.5, 0.3, '🏠 Houses', fontsize=20, ha='center')
    axes5[3].set_title('Layer 5: Full Objects', fontsize=12)
    axes5[3].axis('off')

    plt.suptitle('CNN Feature Hierarchy: Simple → Complex', fontsize=14, y=1.05)
    plt.tight_layout()
    fig5
    return ax, axes5, color, edge_filters, fig5, filt, i, ii, jj, y_offset


@app.cell
def _(mo):
    mo.md(r"""
    ## 9.6 Receptive Field

    The region of input that affects one output neuron.

    $$R = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i=1}^{l-1} s_i$$
    """)
    return


@app.cell
def _(np, plt):
    # Calculate receptive field
    def calc_receptive_field(layers):
        """
        layers: list of (kernel_size, stride) tuples
        """
        rf = 1
        stride_product = 1
        
        for k, s in layers:
            rf = rf + (k - 1) * stride_product
            stride_product *= s
        
        return rf

    # Example: VGG-like network
    vgg_layers = [
        (3, 1), (3, 1), (2, 2),  # Block 1: 2 conv + pool
        (3, 1), (3, 1), (2, 2),  # Block 2
        (3, 1), (3, 1), (3, 1), (2, 2),  # Block 3
        (3, 1), (3, 1), (3, 1), (2, 2),  # Block 4
        (3, 1), (3, 1), (3, 1), (2, 2),  # Block 5
    ]

    # Calculate RF at each layer
    rfs = [1]
    for i in range(1, len(vgg_layers) + 1):
        rf = calc_receptive_field(vgg_layers[:i])
        rfs.append(rf)

    fig6, ax6 = plt.subplots(figsize=(12, 5))

    layer_names = ['Input'] + [f'L{i+1}' for i in range(len(vgg_layers))]
    colors = ['green' if 'pool' not in str(l) else 'red' for l in ['input'] + [l for l in vgg_layers]]

    ax6.bar(range(len(rfs)), rfs, color=['blue'] + ['green' if vgg_layers[i][1] == 1 else 'red' 
                                                     for i in range(len(vgg_layers))])
    ax6.set_xticks(range(len(rfs)))
    ax6.set_xticklabels(layer_names, rotation=45, ha='right')
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Receptive Field (pixels)')
    ax6.set_title('Receptive Field Growth in VGG-like Network\n(Green=Conv, Red=Pool)')
    ax6.grid(True, alpha=0.3, axis='y')

    # Annotate key points
    ax6.annotate(f'Final RF: {rfs[-1]}×{rfs[-1]}', xy=(len(rfs)-1, rfs[-1]), 
                xytext=(len(rfs)-4, rfs[-1]*1.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12, fontweight='bold')

    fig6
    return (
        ax6,
        calc_receptive_field,
        colors,
        fig6,
        layer_names,
        rf,
        rfs,
        vgg_layers,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Component | Purpose |
    |-----------|---------|
    | **Convolution** | Extract local features, parameter sharing |
    | **Stride** | Control downsampling |
    | **Padding** | Control output size |
    | **Pooling** | Reduce dimensions, translation invariance |
    | **Depth** | Learn hierarchical features |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 9.
    - **LeNet**: LeCun et al. (1998)
    - **AlexNet**: Krizhevsky et al. (2012)
    - **VGG**: Simonyan & Zisserman (2015)

    ## Connection to ML Refined Curriculum

    CNNs automate the feature engineering from Week 9:
    - Manual feature extraction → Learned convolutional filters
    - Hierarchical representation learning
    """)
    return


if __name__ == "__main__":
    app.run()

