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
    # Week 6: Regularization and Generalization

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 7

    ---

    ## Learning Objectives

    - Understand the bias-variance tradeoff
    - Master L2 regularization and dropout
    - Learn batch normalization
    - Implement early stopping
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    return np, plt, train_test_split


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.1 The Overfitting Problem

    **Overfitting**: Model memorizes training data but fails on new data.

    - Training loss ↓ but validation loss ↑
    - High variance, low bias
    """)
    return


@app.cell
def _(np, plt, train_test_split):
    # Demonstrate overfitting
    np.random.seed(42)

    # Generate noisy data
    n_samples = 50
    X_demo = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X_demo).ravel()
    y_demo = y_true + np.random.randn(n_samples) * 0.3

    X_train_demo, X_val_demo, y_train_demo, y_val_demo = train_test_split(
        X_demo, y_demo, test_size=0.3, random_state=42)

    # Fit polynomials of different degrees
    from numpy.polynomial import polynomial as P

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

    degrees = [1, 4, 15]
    titles = ['Underfitting (degree=1)', 'Good Fit (degree=4)', 'Overfitting (degree=15)']

    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)

    for ax, degree, title in zip(axes1, degrees, titles):
        # Fit polynomial
        coeffs = np.polyfit(X_train_demo.ravel(), y_train_demo, degree)
        poly = np.poly1d(coeffs)
        
        # Compute errors
        train_error = np.mean((poly(X_train_demo.ravel()) - y_train_demo) ** 2)
        val_error = np.mean((poly(X_val_demo.ravel()) - y_val_demo) ** 2)
        
        # Plot
        ax.scatter(X_train_demo, y_train_demo, c='blue', s=30, label='Train', alpha=0.7)
        ax.scatter(X_val_demo, y_val_demo, c='red', s=30, label='Validation', alpha=0.7)
        ax.plot(X_plot, poly(X_plot), 'g-', linewidth=2, label='Model')
        ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'k--', alpha=0.5, label='True')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title}\nTrain MSE: {train_error:.3f}, Val MSE: {val_error:.3f}')
        ax.legend(fontsize=8)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    fig1
    return (
        P,
        X_demo,
        X_plot,
        X_train_demo,
        X_val_demo,
        ax,
        axes1,
        coeffs,
        degree,
        degrees,
        fig1,
        n_samples,
        poly,
        title,
        titles,
        train_error,
        val_error,
        y_demo,
        y_train_demo,
        y_true,
        y_val_demo,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.2 L2 Regularization (Weight Decay)

    Add penalty on weight magnitudes:
    $$L_{total} = L_{data} + \frac{\lambda}{2} \|W\|^2$$

    **Effect**: Shrinks weights, reduces model complexity
    """)
    return


@app.cell
def _(np, plt):
    # Visualize L2 regularization effect
    np.random.seed(42)

    # Generate polynomial features
    n_train = 20
    X_train_l2 = np.random.uniform(0, 1, n_train)
    y_train_l2 = np.sin(2 * np.pi * X_train_l2) + np.random.randn(n_train) * 0.3

    # Create polynomial features (degree 10)
    degree_l2 = 10
    X_poly = np.column_stack([X_train_l2 ** i for i in range(degree_l2 + 1)])

    # Ridge regression with different lambda
    def ridge_regression(X, y, lambda_reg):
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't regularize bias
        w = np.linalg.solve(X.T @ X + lambda_reg * I, X.T @ y)
        return w

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    lambdas = [0, 0.001, 1.0]
    lambda_names = ['λ=0 (No reg)', 'λ=0.001', 'λ=1.0']

    X_plot_l2 = np.linspace(0, 1, 100)
    X_plot_poly = np.column_stack([X_plot_l2 ** i for i in range(degree_l2 + 1)])

    for ax, lam, name in zip(axes2, lambdas, lambda_names):
        w = ridge_regression(X_poly, y_train_l2, lam)
        y_pred = X_plot_poly @ w
        
        ax.scatter(X_train_l2, y_train_l2, c='blue', s=50, alpha=0.7, label='Data')
        ax.plot(X_plot_l2, y_pred, 'r-', linewidth=2, label='Model')
        ax.plot(X_plot_l2, np.sin(2 * np.pi * X_plot_l2), 'k--', alpha=0.5, label='True')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}\n||w||² = {np.sum(w**2):.2f}')
        ax.legend(fontsize=8)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    fig2
    return (
        X_plot_l2,
        X_plot_poly,
        X_poly,
        X_train_l2,
        ax,
        axes2,
        degree_l2,
        fig2,
        lam,
        lambda_names,
        lambdas,
        n_train,
        name,
        ridge_regression,
        w,
        y_pred,
        y_train_l2,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.3 Dropout

    During training, randomly zero out neurons with probability $p$:
    $$h = \frac{1}{1-p} \cdot \text{mask} \odot \sigma(z)$$

    **Effect**: Creates implicit ensemble of sub-networks
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate dropout
    class DropoutDemo:
        def __init__(self, p=0.5):
            self.p = p
            self.training = True
        
        def forward(self, x):
            if self.training:
                mask = (np.random.rand(*x.shape) > self.p)
                return x * mask / (1 - self.p)
            return x

    # Visualize dropout effect
    np.random.seed(42)
    x = np.ones((1, 10))  # 10 neurons, all active

    fig3, axes3 = plt.subplots(2, 4, figsize=(14, 6))

    dropout = DropoutDemo(p=0.5)

    # Show 8 different dropout masks
    for i, ax in enumerate(axes3.flat):
        np.random.seed(i)
        output = dropout.forward(x.copy())
        
        colors = ['green' if v > 0 else 'red' for v in output[0]]
        ax.bar(range(10), output[0], color=colors)
        ax.axhline(1, color='blue', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 2.5)
        ax.set_title(f'Sample {i+1}: {int(np.sum(output > 0))}/10 active')
        ax.set_xticks(range(10))
        if i >= 4:
            ax.set_xlabel('Neuron')
        if i % 4 == 0:
            ax.set_ylabel('Output')

    plt.suptitle('Dropout (p=0.5): Different Random Masks\nGreen=Active (scaled by 2), Red=Dropped', fontsize=12, y=1.02)
    plt.tight_layout()
    fig3
    return DropoutDemo, ax, axes3, colors, dropout, fig3, i, output, x


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.4 Batch Normalization

    Normalize activations within each mini-batch:
    $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
    $$y = \gamma \hat{x} + \beta$$

    **Benefits**: Faster training, regularization, higher learning rates
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate batch normalization effect
    np.random.seed(42)

    # Simulate activations before BatchNorm
    n_samples_bn = 1000
    mean_shift = 5.0
    scale = 3.0
    activations_before = np.random.randn(n_samples_bn) * scale + mean_shift

    # Apply BatchNorm
    mu = activations_before.mean()
    sigma = activations_before.std()
    activations_after = (activations_before - mu) / (sigma + 1e-5)

    # Apply learned scale and shift (example)
    gamma, beta = 1.5, 0.5
    activations_final = gamma * activations_after + beta

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    axes4[0].hist(activations_before, bins=30, density=True, alpha=0.7, color='blue')
    axes4[0].axvline(mu, color='red', linestyle='--', label=f'mean={mu:.2f}')
    axes4[0].set_xlabel('Activation')
    axes4[0].set_ylabel('Density')
    axes4[0].set_title(f'Before BatchNorm\nμ={mu:.2f}, σ={sigma:.2f}')
    axes4[0].legend()

    axes4[1].hist(activations_after, bins=30, density=True, alpha=0.7, color='green')
    axes4[1].axvline(0, color='red', linestyle='--', label='mean=0')
    axes4[1].set_xlabel('Activation')
    axes4[1].set_ylabel('Density')
    axes4[1].set_title('After Normalization\nμ=0, σ=1')
    axes4[1].legend()

    axes4[2].hist(activations_final, bins=30, density=True, alpha=0.7, color='purple')
    axes4[2].axvline(activations_final.mean(), color='red', linestyle='--', 
                    label=f'mean={activations_final.mean():.2f}')
    axes4[2].set_xlabel('Activation')
    axes4[2].set_ylabel('Density')
    axes4[2].set_title(f'After Scale/Shift (γ={gamma}, β={beta})\nLearnable parameters')
    axes4[2].legend()

    plt.tight_layout()
    fig4
    return (
        activations_after,
        activations_before,
        activations_final,
        axes4,
        beta,
        fig4,
        gamma,
        mean_shift,
        mu,
        n_samples_bn,
        scale,
        sigma,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.5 Early Stopping

    Stop training when validation loss starts increasing:

    1. Monitor validation loss
    2. Save best model
    3. Stop if no improvement for $k$ epochs
    """)
    return


@app.cell
def _(np, plt):
    # Simulate early stopping
    np.random.seed(42)

    epochs_es = 100
    train_loss = 1.0 / (1 + np.arange(epochs_es) * 0.1) + np.random.randn(epochs_es) * 0.02
    val_loss = 1.0 / (1 + np.arange(epochs_es) * 0.08) + 0.02 * np.arange(epochs_es) / epochs_es + np.random.randn(epochs_es) * 0.02

    # Find early stopping point
    patience = 10
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    stop_epoch = epochs_es

    for epoch in range(epochs_es):
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            stop_epoch = epoch
            break

    fig5, ax5 = plt.subplots(figsize=(10, 5))

    ax5.plot(train_loss, 'b-', linewidth=2, label='Training Loss')
    ax5.plot(val_loss, 'r-', linewidth=2, label='Validation Loss')
    ax5.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Model (epoch {best_epoch})')
    ax5.axvline(stop_epoch, color='orange', linestyle='--', linewidth=2, label=f'Early Stop (epoch {stop_epoch})')

    ax5.fill_between(range(stop_epoch, epochs_es), 0, max(val_loss), alpha=0.2, color='red', label='Overfitting region')

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title(f'Early Stopping (patience={patience})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig5
    return (
        ax5,
        best_epoch,
        best_val_loss,
        epoch,
        epochs_es,
        fig5,
        patience,
        patience_counter,
        stop_epoch,
        train_loss,
        val_loss,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.6 Data Augmentation

    Artificially expand training data by applying transformations.

    | Type | Transformations |
    |------|-----------------|
    | Geometric | Rotation, flip, crop, scale |
    | Color | Brightness, contrast, hue |
    | Noise | Gaussian noise, blur |
    | Mixing | Mixup, CutMix |
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate data augmentation on a simple 2D example
    np.random.seed(42)

    # Original point
    original_point = np.array([0.5, 0.5])

    # Augmentation functions
    def rotate(point, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return R @ (point - 0.5) + 0.5

    def scale(point, factor):
        return 0.5 + (point - 0.5) * factor

    def translate(point, offset):
        return point + offset

    def add_noise(point, std):
        return point + np.random.randn(2) * std

    # Generate augmented samples
    n_aug = 50
    augmented_points = []
    for _ in range(n_aug):
        p = original_point.copy()
        # Random rotation
        p = rotate(p, np.random.uniform(-0.5, 0.5))
        # Random scale
        p = scale(p, np.random.uniform(0.8, 1.2))
        # Random translation
        p = translate(p, np.random.uniform(-0.1, 0.1, 2))
        # Random noise
        p = add_noise(p, 0.02)
        augmented_points.append(p)

    augmented_points = np.array(augmented_points)

    # Visualize
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    axes6[0].scatter([original_point[0]], [original_point[1]], s=200, c='red', marker='*', label='Original')
    axes6[0].set_xlim(0, 1)
    axes6[0].set_ylim(0, 1)
    axes6[0].set_aspect('equal')
    axes6[0].set_title('Original Training Point')
    axes6[0].legend()
    axes6[0].grid(True, alpha=0.3)

    # Augmented
    axes6[1].scatter(augmented_points[:, 0], augmented_points[:, 1], s=50, c='blue', alpha=0.5, label='Augmented')
    axes6[1].scatter([original_point[0]], [original_point[1]], s=200, c='red', marker='*', label='Original')
    axes6[1].set_xlim(0, 1)
    axes6[1].set_ylim(0, 1)
    axes6[1].set_aspect('equal')
    axes6[1].set_title(f'After Augmentation ({n_aug} samples)')
    axes6[1].legend()
    axes6[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig6
    return (
        add_noise,
        augmented_points,
        axes6,
        fig6,
        n_aug,
        original_point,
        p,
        rotate,
        scale,
        translate,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.7 Mixup: Advanced Augmentation

    Create virtual training examples by mixing:
    $$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
    $$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

    Where $\lambda \sim \text{Beta}(\alpha, \alpha)$
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate Mixup
    def visualize_mixup():
        np.random.seed(42)
        
        # Two sample "images" (simplified as 1D signals)
        x1 = np.sin(np.linspace(0, 2*np.pi, 50))
        x2 = np.cos(np.linspace(0, 2*np.pi, 50))
        y1, y2 = np.array([1, 0]), np.array([0, 1])  # One-hot labels
        
        # Different lambda values
        lambdas_mix = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        for ax, lam in zip(axes, lambdas_mix):
            x_mixed = lam * x1 + (1 - lam) * x2
            y_mixed = lam * y1 + (1 - lam) * y2
            
            ax.plot(x_mixed, 'purple', linewidth=2)
            ax.set_title(f'λ={lam}\ny=[{y_mixed[0]:.1f}, {y_mixed[1]:.1f}]')
            ax.set_ylim(-1.5, 1.5)
        
        plt.suptitle('Mixup: Interpolating Between Two Samples', fontsize=12, y=1.05)
        plt.tight_layout()
        return fig

    fig7 = visualize_mixup()
    fig7
    return fig7, visualize_mixup


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Technique | Mechanism | When to Use |
    |-----------|-----------|-------------|
    | **L2 Regularization** | Penalize large weights | Always (default) |
    | **Dropout** | Random neuron dropping | Fully connected layers |
    | **BatchNorm** | Normalize activations | CNNs |
    | **Early Stopping** | Stop at best validation | Always |
    | **Data Augmentation** | Transform training data | Limited data |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 7.
    - **Dropout**: Srivastava et al. (2014). "Dropout: A simple way to prevent overfitting."
    - **BatchNorm**: Ioffe & Szegedy (2015). "Batch Normalization."

    ## Connection to ML Refined Curriculum

    Regularization prevents overfitting for:
    - All supervised learning (Weeks 4-8)
    - Feature selection (Week 9)
    """)
    return


if __name__ == "__main__":
    app.run()

