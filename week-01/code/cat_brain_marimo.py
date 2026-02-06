import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return mo, nn, np, pd, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    # 🐱 Cat Brain Model — Interactive PyTorch Tutorial

    **From "Math and Architectures of Deep Learning" - Chapter 1**

    This interactive notebook demonstrates a complete machine learning pipeline:
    1. **Data Preparation** — Generate and normalize training data
    2. **Model Architecture** — Define a simple linear model
    3. **Training** — Iteratively adjust weights to minimize loss
    4. **Inference** — Apply the trained model to new inputs

    Use the controls below to experiment with different parameters and see how
    they affect training and predictions!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Section 1: Data Preparation
    """)
    return


@app.cell
def _(mo):
    seed_slider = mo.ui.slider(start=1, stop=100, step=1, value=42, label="🎲 Random Seed", show_value=True)
    n_train_slider = mo.ui.slider(start=10, stop=500, step=10, value=200, label="📈 Training Samples", show_value=True)
    n_test_slider = mo.ui.slider(start=10, stop=100, step=10, value=50, label="📉 Test Samples", show_value=True)
    noise_slider = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.05, label="🔊 Noise Level (σ)", show_value=True)

    mo.md(
        f"""
        ### Data Generation Parameters

        Adjust these to see how data quantity and quality affect learning:

        | Parameter | Control |
        |-----------|---------|
        | Random Seed | {seed_slider} |
        | Training Samples | {n_train_slider} |
        | Test Samples | {n_test_slider} |
        | Noise Level | {noise_slider} |

        **Try:** Set noise to 0.3+ and see how the model struggles!
        """
    )
    return n_test_slider, n_train_slider, noise_slider, seed_slider


@app.cell
def _(np, torch):
    def generate_training_data(n_samples, noise_sigma, rng):
        """
        Generate synthetic training data for the cat brain model.

        Ground truth: threat = (hardness + sharpness - 1) / sqrt(2)
        This is the signed distance from the line x0 + x1 = 1

        Args:
            n_samples: Number of data points to generate
            noise_sigma: Standard deviation of Gaussian noise
            rng: NumPy random generator for reproducibility
        """
        # Random hardness and sharpness values in [0, 1]
        _hardness = rng.random(n_samples)
        _sharpness = rng.random(n_samples)

        # Ground truth threat score (Equation 1.4)
        # y = (x0 + x1 - 1) / sqrt(2)
        _threat_gt = (_hardness + _sharpness - 1) / np.sqrt(2)

        # Add noise to make it realistic
        _threat_gt += rng.standard_normal(n_samples) * noise_sigma

        # Stack into tensors
        X = torch.tensor(np.column_stack([_hardness, _sharpness]), dtype=torch.float32)
        y = torch.tensor(_threat_gt.reshape(-1, 1), dtype=torch.float32)

        return X, y
    return (generate_training_data,)


@app.cell
def _(
    generate_training_data,
    mo,
    n_test_slider,
    n_train_slider,
    noise_slider,
    np,
    seed_slider,
    torch,
):
    # Set seeds for reproducibility
    _seed = int(seed_slider.value)
    torch.manual_seed(_seed)
    _rng = np.random.default_rng(_seed)

    # Generate datasets
    X_train, y_train = generate_training_data(
        int(n_train_slider.value),
        noise_slider.value,
        _rng
    )
    X_test, y_test = generate_training_data(
        int(n_test_slider.value),
        noise_slider.value,
        _rng
    )

    mo.md(
        f"""
        ### Generated Data

        - **Training set:** {X_train.shape[0]} samples
        - **Test set:** {X_test.shape[0]} samples
        - **Features:** hardness (x₀), sharpness (x₁)
        - **Target:** threat score (y)

        Sample data point:
        - Input: `(hardness={X_train[0, 0].item():.3f}, sharpness={X_train[0, 1].item():.3f})`
        - Output: `threat={y_train[0].item():.3f}`
        """
    )
    return X_train, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 🧠 Section 2: Model Architecture

    We use a simple **linear model**:

    $$y = w_0 \cdot x_0 + w_1 \cdot x_1 + b = \mathbf{w}^T \mathbf{x} + b$$

    Where:
    - $x_0$ = hardness, $x_1$ = sharpness
    - $w_0, w_1$ = weights (how much each input contributes)
    - $b$ = bias (constant offset)
    """)
    return


@app.cell
def _(nn):
    class CatBrainModel(nn.Module):
        """
        Simple linear model: y = w0*x0 + w1*x1 + b

        This is Equation 1.3 from the textbook:
        y(x0, x1) = w0*x0 + w1*x1 + b = w^T * x + b
        """

        def __init__(self):
            super().__init__()
            # nn.Linear(in_features, out_features) creates:
            # - weight matrix of shape [out_features, in_features]
            # - bias vector of shape [out_features]
            self.linear = nn.Linear(2, 1)

            # Initialize with random values (as in Algorithm 1.1)
            nn.init.normal_(self.linear.weight, mean=0, std=0.5)
            nn.init.zeros_(self.linear.bias)

        def forward(self, x):
            """Forward pass: compute y = w^T * x + b"""
            return self.linear(x)

        def get_parameters(self):
            """Return current parameter values for inspection"""
            _w = self.linear.weight.detach().numpy().flatten()
            _b = self.linear.bias.detach().numpy()[0]
            return _w[0], _w[1], _b
    return (CatBrainModel,)


@app.cell
def _(mo):
    mo.md("""
    ## 🏋️ Section 3: Training
    """)
    return


@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(start=50, stop=1000, step=50, value=500, label="🔄 Epochs", show_value=True)
    lr_slider = mo.ui.slider(start=0.001, stop=1.0, step=0.001, value=0.1, label="📐 Learning Rate", show_value=True)
    optimizer_dropdown = mo.ui.dropdown(
        options=["SGD", "Adam"],
        value="SGD",
        label="⚙️ Optimizer"
    )

    mo.md(
        f"""
        ### Training Hyperparameters

        | Parameter | Control |
        |-----------|---------|
        | Epochs | {epochs_slider} |
        | Learning Rate | {lr_slider} |
        | Optimizer | {optimizer_dropdown} |

        **Experiments to try:**
        - Very small learning rate (0.001): Watch slow convergence
        - Very large learning rate (1.0): Watch instability
        - Few epochs (50): Watch underfitting
        """
    )
    return epochs_slider, lr_slider, optimizer_dropdown


@app.cell
def _(
    CatBrainModel,
    X_train,
    epochs_slider,
    lr_slider,
    mo,
    nn,
    np,
    optimizer_dropdown,
    seed_slider,
    torch,
    y_train,
):
    # Reset seed for reproducible weight initialization
    torch.manual_seed(int(seed_slider.value))

    # Create fresh model
    model = CatBrainModel()
    _w0_init, _w1_init, _b_init = model.get_parameters()

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer selection
    if optimizer_dropdown.value == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_slider.value)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_slider.value)

    # Training history
    history = {"loss": [], "w0": [], "w1": [], "b": []}

    # Training loop
    _num_epochs = int(epochs_slider.value)
    for _epoch in range(_num_epochs):
        # Forward pass
        _y_pred = model(X_train)

        # Compute loss
        _loss = loss_fn(_y_pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        _loss.backward()

        # Update parameters
        optimizer.step()

        # Record history
        _w0, _w1, _b = model.get_parameters()
        history["loss"].append(_loss.item())
        history["w0"].append(_w0)
        history["w1"].append(_w1)
        history["b"].append(_b)

    # Final parameters
    w0_final, w1_final, b_final = model.get_parameters()

    # Theoretical optimal
    _w_opt = 1 / np.sqrt(2)
    _b_opt = -1 / np.sqrt(2)

    mo.md(
        f"""
        ### Training Results

        | Parameter | Initial | Final | Optimal |
        |-----------|---------|-------|---------|
        | w₀ | {_w0_init:.4f} | {w0_final:.4f} | {_w_opt:.4f} |
        | w₁ | {_w1_init:.4f} | {w1_final:.4f} | {_w_opt:.4f} |
        | b  | {_b_init:.4f} | {b_final:.4f} | {_b_opt:.4f} |

        **Final Loss:** {history['loss'][-1]:.6f}
        """
    )
    return history, model


@app.cell
def _(mo):
    mo.md("""
    ## 📈 Section 4: Training Visualization
    """)
    return


@app.cell
def _(X_train, history, model, np, plt, torch, y_train):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Loss over time
    ax1 = axes[0, 0]
    ax1.plot(history["loss"], "b-", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training Loss Over Time")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter convergence
    ax2 = axes[0, 1]
    ax2.plot(history["w0"], "r-", label="w₀", linewidth=2)
    ax2.plot(history["w1"], "g-", label="w₁", linewidth=2)
    ax2.plot(history["b"], "b-", label="b", linewidth=2)
    ax2.axhline(y=1/np.sqrt(2), color="orange", linestyle="--", alpha=0.7, label="optimal w")
    ax2.axhline(y=-1/np.sqrt(2), color="purple", linestyle="--", alpha=0.7, label="optimal b")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("Parameter Convergence")
    ax2.legend(loc="right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Feature space with decision boundary
    ax3 = axes[1, 0]
    with torch.no_grad():
        _predictions = model(X_train).numpy().flatten()
    _scatter = ax3.scatter(
        X_train[:, 0].numpy(),
        X_train[:, 1].numpy(),
        c=_predictions,
        cmap="RdYlGn_r",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5
    )
    plt.colorbar(_scatter, ax=ax3, label="Threat Score")

    # Decision boundary
    _w0, _w1, _b = model.get_parameters()
    _x0_line = np.array([0, 1])
    _x1_line = (-_w0 * _x0_line - _b) / _w1
    ax3.plot(_x0_line, _x1_line, "k-", linewidth=2, label="Decision boundary (y=0)")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("X₀ (Hardness)")
    ax3.set_ylabel("X₁ (Sharpness)")
    ax3.set_title("Feature Space with Decision Boundary")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_aspect("equal")

    # Plot 4: Predicted vs Actual
    ax4 = axes[1, 1]
    with torch.no_grad():
        _y_pred = model(X_train).numpy().flatten()
    _y_actual = y_train.numpy().flatten()
    ax4.scatter(_y_actual, _y_pred, alpha=0.5, s=30)
    ax4.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect prediction")
    ax4.set_xlabel("Actual Threat Score")
    ax4.set_ylabel("Predicted Threat Score")
    ax4.set_title("Predicted vs Actual")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🔮 Section 5: Inference
    """)
    return


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(start=0.05, stop=0.5, step=0.01, value=0.2, label="🎯 Decision Threshold", show_value=True)

    mo.md(
        f"""
        ### Decision Threshold

        The cat decides what to do based on the threat score:
        - **y > threshold**: Run away! 🏃
        - **|y| ≤ threshold**: Ignore 😐
        - **y < -threshold**: Approach and purr 😻

        {threshold_slider}
        """
    )
    return (threshold_slider,)


@app.cell
def _(mo):
    custom_name_input = mo.ui.text(value="Rock", label="Object Name")
    custom_hardness_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.7, label="Hardness", show_value=True)
    custom_sharpness_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.3, label="Sharpness", show_value=True)

    mo.md(
        f"""
        ### 🧪 Test Your Own Object

        Create a custom object to see how the cat reacts:

        | Property | Value |
        |----------|-------|
        | Name | {custom_name_input} |
        | Hardness | {custom_hardness_slider} |
        | Sharpness | {custom_sharpness_slider} |
        """
    )
    return custom_hardness_slider, custom_name_input, custom_sharpness_slider


@app.cell
def _(
    custom_hardness_slider,
    custom_name_input,
    custom_sharpness_slider,
    mo,
    model,
    pd,
    threshold_slider,
    torch,
):
    def _make_decision(threat_score, threshold):
        if threat_score > threshold:
            return "Run away! 🏃"
        elif threat_score < -threshold:
            return "Approach and purr 😻"
        else:
            return "Ignore 😐"

    # Predefined test objects + custom
    _test_objects = [
        ("Pillow", 0.1, 0.1),
        ("Book", 0.5, 0.5),
        ("Knife", 0.9, 0.95),
        ("Blanket", 0.15, 0.2),
        ("Cactus", 0.85, 0.9),
        (
            custom_name_input.value or "Custom",
            custom_hardness_slider.value,
            custom_sharpness_slider.value
        ),
    ]

    _rows = []
    for _name, _h, _s in _test_objects:
        _x = torch.tensor([[_h, _s]], dtype=torch.float32)
        with torch.no_grad():
            _threat = model(_x).item()
        _rows.append({
            "Object": _name,
            "Hardness": f"{_h:.2f}",
            "Sharpness": f"{_s:.2f}",
            "Threat Score": f"{_threat:+.4f}",
            "Decision": _make_decision(_threat, threshold_slider.value),
        })

    inference_df = pd.DataFrame(_rows)

    mo.md(
        f"""
        ### Inference Results

        {mo.ui.table(inference_df, selection=None)}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📝 Summary

    ### Key Takeaways

    1. **Model Architecture:** $y = w_0 \cdot x_0 + w_1 \cdot x_1 + b$
       - Simple weighted sum with bias
       - Parameters: $w_0, w_1$ (weights), $b$ (bias)

    2. **Training Process:**
       - Initialize parameters randomly
       - Iteratively adjust to minimize MSE loss
       - Gradient descent finds optimal weights

    3. **Inference:**
       - Apply trained model to new inputs
       - Threshold the output for classification

    4. **Geometric Interpretation:**
       - Decision boundary is a line in 2D feature space
       - Points above/below the line get different classifications

    ### Exercises to Try

    1. **Parameter Sensitivity:** Try learning rate = 0.001 vs 1.0
    2. **Data Quantity:** Compare 10 samples vs 500 samples
    3. **Noise Analysis:** Increase noise to 0.3 and observe degradation
    4. **Optimal Values:** Verify that $w_0 = w_1 = 1/\sqrt{2} \approx 0.707$ and $b = -1/\sqrt{2} \approx -0.707$
    """)
    return


if __name__ == "__main__":
    app.run()
