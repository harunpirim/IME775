# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "torch==2.10.0",
# ]
# ///
"""
IME 775: XOR with PyTorch — Clean Exploratory Notebook
=======================================================
A focused, interactive marimo notebook that teaches the complete
PyTorch training pipeline through the XOR problem.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 8 — Training Neural Networks
Topics: Forward Pass, MSE Loss, Backpropagation, Gradient Descent, Training Loop
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    return nn, np, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    # IME 775: XOR with PyTorch — Exploratory Lab

    ## Learning Objectives

    1. Trace a forward pass through a small network with concrete numbers
    2. Compute MSE loss and understand what the gradient tells us
    3. Run backpropagation and inspect every gradient PyTorch computes
    4. Perform a gradient descent step and verify the loss decreases
    5. Train a full XOR network with tunable hyperparameters

    ---

    | Section | Topic |
    |---------|-------|
    | 1 | Step-by-Step Forward Pass |
    | 2 | MSE Loss & Its Gradient |
    | 3 | Backpropagation — Inspecting All Gradients |
    | 4 | One Gradient Descent Step |
    | 5 | Full XOR Training Loop |
    """)
    return


# ═══════════════════════════════════════════════════════════════
# Section 1 — Step-by-Step Forward Pass
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Step-by-Step Forward Pass

    We build a **2 → 4 → 1** network (sigmoid activations) and trace every
    intermediate value.  This is exactly what `model(X)` does internally.

    $$\vec{z}^{(l)} = W^{(l)}\,\vec{a}^{(l-1)} + \vec{b}^{(l)}, \qquad
      \vec{a}^{(l)} = \sigma\!\bigl(\vec{z}^{(l)}\bigr)$$
    """)
    return


@app.cell
def _(mo, nn, torch):
    _model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    torch.manual_seed(0)
    for _p in _model.parameters():
        nn.init.uniform_(_p, -1, 1)

    _x = torch.tensor([[1.0, 0.0]])

    _z0 = _model[0](_x)
    _a0 = _model[1](_z0)
    _z1 = _model[2](_a0)
    _y  = _model[3](_z1)

    _W0 = _model[0].weight.data
    _b0 = _model[0].bias.data
    _W1 = _model[2].weight.data
    _b1 = _model[2].bias.data

    mo.md(
        f"""
        ### Network Weights (randomly initialised)

        **Layer 0** weights ({_W0.shape[0]}×{_W0.shape[1]}):

        | | x₁ weight | x₂ weight | bias |
        |---|---|---|---|
        | neuron 0 | {_W0[0,0]:.4f} | {_W0[0,1]:.4f} | {_b0[0]:.4f} |
        | neuron 1 | {_W0[1,0]:.4f} | {_W0[1,1]:.4f} | {_b0[1]:.4f} |
        | neuron 2 | {_W0[2,0]:.4f} | {_W0[2,1]:.4f} | {_b0[2]:.4f} |
        | neuron 3 | {_W0[3,0]:.4f} | {_W0[3,1]:.4f} | {_b0[3]:.4f} |

        **Layer 1** weights ({_W1.shape[0]}×{_W1.shape[1]}):

        | a₀ | a₁ | a₂ | a₃ | bias |
        |---|---|---|---|---|
        | {_W1[0,0]:.4f} | {_W1[0,1]:.4f} | {_W1[0,2]:.4f} | {_W1[0,3]:.4f} | {_b1[0]:.4f} |

        ---

        ### Forward Pass Trace for input x = (1.0, 0.0)

        **Layer 0 — pre-activation z⁰ = W⁰·x + b⁰:**

        | neuron | z | σ(z) = a |
        |--------|---|----------|
        | 0 | {_z0[0,0]:.4f} | {_a0[0,0]:.4f} |
        | 1 | {_z0[0,1]:.4f} | {_a0[0,1]:.4f} |
        | 2 | {_z0[0,2]:.4f} | {_a0[0,2]:.4f} |
        | 3 | {_z0[0,3]:.4f} | {_a0[0,3]:.4f} |

        **Layer 1 — output:**

        | z¹ | y = σ(z¹) |
        |---|---|
        | {_z1[0,0]:.4f} | **{_y[0,0]:.4f}** |

        PyTorch `model(x)` gives the same answer: **{_model(_x)[0,0]:.4f}** ✓
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# Section 2 — MSE Loss & Its Gradient
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. MSE Loss & Its Gradient

    For a single example with target $\bar{y}$:

    $$\ell = \frac{1}{2}(\bar{y} - y)^2, \qquad
      \frac{\partial \ell}{\partial y} = -(\ \bar{y} - y)$$

    The gradient points **away** from the target.
    Gradient descent (subtract gradient) pushes the prediction **toward** the target.
    """)
    return


@app.cell
def _(mo, np, plt, torch):
    _preds = torch.tensor([0.2, 0.5, 0.8, 0.95], requires_grad=True)
    _target = 1.0
    _losses = 0.5 * (_target - _preds) ** 2
    _total = _losses.sum()
    _total.backward()

    _yc = np.linspace(-0.2, 1.5, 200)
    _lc = 0.5 * (_target - _yc) ** 2

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b']
    _ax1.bar([f'y={p:.1f}' for p in _preds.detach().numpy()],
             _losses.detach().numpy(), color=_colors, edgecolor='white', linewidth=1.5)
    for _j, _l in enumerate(_losses.detach().numpy()):
        _ax1.text(_j, _l + 0.005, f'{_l:.4f}', ha='center', fontweight='bold', fontsize=10)
    _ax1.set_title('Per-prediction loss  ℓ = ½(ȳ − y)²', fontweight='bold')
    _ax1.set_ylabel('Loss')

    _ax2.plot(_yc, _lc, 'b-', alpha=0.4, linewidth=2)
    for _j, (_p, _g) in enumerate(zip(_preds.detach().numpy(), _preds.grad.numpy())):
        _lv = float(0.5 * (_target - _p) ** 2)
        _ax2.scatter([_p], [_lv], c=_colors[_j], s=100, zorder=5, edgecolors='white', linewidth=1.5)
        _ax2.annotate(f'∂ℓ/∂y = {_g:.2f}', (_p, _lv + 0.02),
                     fontsize=9, ha='center', color=_colors[_j], fontweight='bold')
    _ax2.axvline(_target, color='red', ls='--', alpha=0.5, label=f'target ȳ = {_target}')
    _ax2.set_title('Loss curve & gradients', fontweight='bold')
    _ax2.set_xlabel('y (prediction)')
    _ax2.set_ylabel('Loss')
    _ax2.legend()
    _ax2.grid(alpha=0.2)
    _fig.suptitle('MSE Loss — gradient always points away from target', fontsize=13, fontweight='bold')
    plt.tight_layout()

    mo.md(
        f"""
        ### Results

        | Prediction y | Target ȳ | Loss ½(ȳ−y)² | ∂ℓ/∂y = −(ȳ−y) | Gradient says |
        |---|---|---|---|---|
        | 0.20 | 1.0 | {_losses[0].item():.4f} | {_preds.grad[0].item():.4f} | increase y ↑ |
        | 0.50 | 1.0 | {_losses[1].item():.4f} | {_preds.grad[1].item():.4f} | increase y ↑ |
        | 0.80 | 1.0 | {_losses[2].item():.4f} | {_preds.grad[2].item():.4f} | increase y ↑ |
        | 0.95 | 1.0 | {_losses[3].item():.4f} | {_preds.grad[3].item():.4f} | increase y ↑ |

        Subtracting the gradient moves the prediction toward the target — this is gradient descent.
        """
    )
    _fig
    return


# ═══════════════════════════════════════════════════════════════
# Section 3 — Backpropagation — Inspecting All Gradients
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Backpropagation — Inspecting All Gradients

    We build a **1 → 2 → 1** network (same as the lecture notes workout),
    run one forward pass, call `loss.backward()`, and then print every
    gradient that PyTorch computed.

    We also compute each gradient **by hand** and verify they match.
    """)
    return


@app.cell
def _(mo, nn, torch):
    _net = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        _net[0].weight.copy_(torch.tensor([[0.5], [-0.3]]))
        _net[0].bias.copy_(torch.tensor([0.1, -0.1]))
        _net[2].weight.copy_(torch.tensor([[0.4, 0.6]]))
        _net[2].bias.copy_(torch.tensor([0.2]))

    _x = torch.tensor([[1.0]])
    _ybar = torch.tensor([[1.0]])

    _z0 = _net[0](_x)
    _a0 = _net[1](_z0)
    _z1 = _net[2](_a0)
    _y  = _net[3](_z1)

    _loss = 0.5 * (_ybar - _y) ** 2
    _loss.backward()

    _gW0 = _net[0].weight.grad.clone()
    _gb0 = _net[0].bias.grad.clone()
    _gW1 = _net[2].weight.grad.clone()
    _gb1 = _net[2].bias.grad.clone()

    _d1 = -(_ybar.item() - _y.item()) * _y.item() * (1 - _y.item())
    _a0_np = _a0.detach()
    _d00 = _d1 * 0.4 * _a0_np[0, 0].item() * (1 - _a0_np[0, 0].item())
    _d10 = _d1 * 0.6 * _a0_np[0, 1].item() * (1 - _a0_np[0, 1].item())

    mo.md(
        f"""
        ### Forward Pass

        | Variable | Value |
        |----------|-------|
        | x | 1.0 |
        | z₀⁰ = 0.5·1.0 + 0.1 | **{_z0[0,0].item():.4f}** |
        | z₁⁰ = −0.3·1.0 + (−0.1) | **{_z0[0,1].item():.4f}** |
        | a₀⁰ = σ(z₀⁰) | **{_a0[0,0].item():.4f}** |
        | a₁⁰ = σ(z₁⁰) | **{_a0[0,1].item():.4f}** |
        | z¹ = 0.4·a₀ + 0.6·a₁ + 0.2 | **{_z1[0,0].item():.4f}** |
        | y = σ(z¹) | **{_y[0,0].item():.4f}** |
        | ℓ = ½(1.0 − y)² | **{_loss.item():.6f}** |

        ### Backward Pass — Manual vs PyTorch

        | Gradient | Manual formula | Manual value | PyTorch `.grad` | Match? |
        |----------|---------------|-------------|-----------------|--------|
        | δ¹ = −(ȳ−y)·σ'(z¹) | | {_d1:.6f} | — | — |
        | ∂ℓ/∂v₀ = δ¹·a₀ | {_d1:.4f}·{_a0_np[0,0].item():.4f} | {_d1*_a0_np[0,0].item():.6f} | {_gW1[0,0].item():.6f} | {'✓' if abs(_d1*_a0_np[0,0].item() - _gW1[0,0].item()) < 1e-4 else '✗'} |
        | ∂ℓ/∂v₁ = δ¹·a₁ | {_d1:.4f}·{_a0_np[0,1].item():.4f} | {_d1*_a0_np[0,1].item():.6f} | {_gW1[0,1].item():.6f} | {'✓' if abs(_d1*_a0_np[0,1].item() - _gW1[0,1].item()) < 1e-4 else '✗'} |
        | ∂ℓ/∂b¹ = δ¹ | | {_d1:.6f} | {_gb1[0].item():.6f} | {'✓' if abs(_d1 - _gb1[0].item()) < 1e-4 else '✗'} |
        | ∂ℓ/∂w₀₀ = δ₀⁰·x | {_d00:.6f}·1.0 | {_d00:.6f} | {_gW0[0,0].item():.6f} | {'✓' if abs(_d00 - _gW0[0,0].item()) < 1e-4 else '✗'} |
        | ∂ℓ/∂w₁₀ = δ₁⁰·x | {_d10:.6f}·1.0 | {_d10:.6f} | {_gW0[1,0].item():.6f} | {'✓' if abs(_d10 - _gW0[1,0].item()) < 1e-4 else '✗'} |

        **All match** — `loss.backward()` computes the same gradients as hand-derived backpropagation.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# Section 4 — One Gradient Descent Step
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. One Gradient Descent Step

    Using the gradients from Section 3, apply the update rule
    $w \leftarrow w - r \cdot \frac{\partial \ell}{\partial w}$
    with learning rate $r = 1.0$, then re-run the forward pass to verify the loss went down.
    """)
    return


@app.cell
def _(mo, nn, torch):
    _r = 1.0

    _net = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        _net[0].weight.copy_(torch.tensor([[0.5], [-0.3]]))
        _net[0].bias.copy_(torch.tensor([0.1, -0.1]))
        _net[2].weight.copy_(torch.tensor([[0.4, 0.6]]))
        _net[2].bias.copy_(torch.tensor([0.2]))

    _x = torch.tensor([[1.0]])
    _ybar = torch.tensor([[1.0]])
    _y_old = _net(_x)
    _loss_old = 0.5 * (_ybar - _y_old) ** 2
    _loss_old.backward()

    _old_w = {n: p.data.clone() for n, p in _net.named_parameters()}
    _grads = {n: p.grad.clone() for n, p in _net.named_parameters()}

    with torch.no_grad():
        for _p in _net.parameters():
            _p -= _r * _p.grad

    _net.zero_grad()
    _y_new = _net(_x)
    _loss_new = 0.5 * (_ybar - _y_new) ** 2

    _rows = []
    for _name in _old_w:
        _o = _old_w[_name].flatten()
        _g = _grads[_name].flatten()
        _n = (_o - _r * _g)
        for _j in range(len(_o)):
            _rows.append(
                f"| {_name}[{_j}] | {_o[_j].item():.4f} | {_g[_j].item():.6f} "
                f"| {-_r*_g[_j].item():+.6f} | {_n[_j].item():.4f} |"
            )

    _table = "\n        ".join(_rows)

    mo.md(
        f"""
        ### Parameter Update (r = {_r})

        | Parameter | Old | Gradient | −r·grad | New |
        |-----------|-----|----------|---------|-----|
        {_table}

        ### Did it improve?

        | | Before | After |
        |---|--------|-------|
        | Prediction y | {_y_old.item():.4f} | {_y_new.item():.4f} |
        | Loss ½(ȳ−y)² | {_loss_old.item():.6f} | {_loss_new.item():.6f} |
        | | | {'✅ Loss decreased!' if _loss_new.item() < _loss_old.item() else '⚠️ Loss increased'} |

        One gradient descent step moved the prediction from {_y_old.item():.4f} → {_y_new.item():.4f}
        (target is 1.0), and the loss dropped.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# Section 5 — Full XOR Training Loop
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Full XOR Training Loop

    Train a neural network to learn XOR — the simplest problem that a
    single-layer network **cannot** solve (it is not linearly separable).

    | Input | Target |
    |-------|--------|
    | (0, 0) | 0 |
    | (0, 1) | 1 |
    | (1, 0) | 1 |
    | (1, 1) | 0 |

    Adjust hyperparameters and click **Train** to run.
    """)
    return


@app.cell
def _(mo):
    xor_form = (
        mo.md(
            """
            **Learning rate:** {lr}

            **Hidden neurons:** {hidden}

            **Epochs:** {epochs}

            **Activation:** {activation}

            **Random seed:** {seed}
            """
        )
        .batch(
            lr=mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0,
                            label="Learning rate"),
            hidden=mo.ui.slider(start=2, stop=16, step=1, value=4,
                                label="Hidden neurons"),
            epochs=mo.ui.slider(start=1000, stop=20000, step=1000, value=10000,
                                label="Epochs"),
            activation=mo.ui.dropdown(options={"Sigmoid": "Sigmoid", "Tanh": "Tanh"},
                                       value="Sigmoid", label="Activation"),
            seed=mo.ui.number(value=42, start=0, stop=999, step=1,
                              label="Random seed"),
        )
        .form(submit_button_label="Train Network")
    )
    xor_form
    return (xor_form,)


@app.cell
def _(mo, nn, np, plt, torch, xor_form):
    mo.stop(xor_form.value is None,
            mo.md("*Adjust hyperparameters above and click **Train Network**.*"))

    _cfg = xor_form.value
    _lr = _cfg["lr"]
    _h = int(_cfg["hidden"])
    _n_ep = int(_cfg["epochs"])
    _act_str = _cfg["activation"]
    _seed = int(_cfg["seed"])

    torch.manual_seed(_seed)

    def _make_act():
        return nn.Sigmoid() if _act_str == "Sigmoid" else nn.Tanh()

    _model = nn.Sequential(
        nn.Linear(2, _h), _make_act(),
        nn.Linear(_h, 1), nn.Sigmoid()
    )

    _X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    _loss_fn = nn.MSELoss()
    _opt = torch.optim.SGD(_model.parameters(), lr=_lr)

    _losses = []
    for _ep in range(_n_ep):
        _yp = _model(_X)
        _loss = _loss_fn(_yp, _Y)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        if _ep % max(1, _n_ep // 500) == 0 or _ep == _n_ep - 1:
            _losses.append((_ep, _loss.item()))

    with torch.no_grad():
        _fp = _model(_X)

    # --- plots ---
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _ep_list = [e for e, _ in _losses]
    _lv_list = [v for _, v in _losses]
    _ax1.plot(_ep_list, _lv_list, 'b-', linewidth=1.5)
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('MSE Loss')
    _ax1.set_title(f'Training Loss ({_act_str}, lr={_lr}, h={_h})', fontweight='bold')
    if min(_lv_list) > 0:
        _ax1.set_yscale('log')
    _ax1.grid(alpha=0.3)
    _ax1.axhline(0.01, color='green', ls='--', alpha=0.5, label='0.01 threshold')
    _ax1.legend()

    _xx = np.linspace(-0.5, 1.5, 100)
    _yy = np.linspace(-0.5, 1.5, 100)
    _XX, _YY = np.meshgrid(_xx, _yy)
    _grid = torch.tensor(np.c_[_XX.ravel(), _YY.ravel()], dtype=torch.float32)
    with torch.no_grad():
        _ZZ = _model(_grid).numpy().reshape(_XX.shape)
    _ax2.contourf(_XX, _YY, _ZZ, levels=20, cmap='RdBu_r', alpha=0.8)
    _ax2.contour(_XX, _YY, _ZZ, levels=[0.5], colors='black', linewidths=2)
    _cxor = ['#3b82f6' if _Y[_j] == 0 else '#ef4444' for _j in range(4)]
    _ax2.scatter(_X[:, 0].numpy(), _X[:, 1].numpy(), c=_cxor, s=200,
                 edgecolors='white', linewidth=2, zorder=5)
    for _j in range(4):
        _ax2.annotate(f'{_fp[_j].item():.3f}',
                     (_X[_j, 0].item(), _X[_j, 1].item()),
                     textcoords="offset points", xytext=(15, -5),
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    _ax2.set_title('Decision Boundary', fontweight='bold')
    _ax2.set_xlabel('x₁')
    _ax2.set_ylabel('x₂')

    _fig.suptitle(f"XOR Training: {_act_str} | lr={_lr} | {_n_ep} epochs",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    xor_results = {
        "preds": [_fp[_j].item() for _j in range(4)],
        "final_loss": _losses[-1][1],
        "n_params": sum(_p.numel() for _p in _model.parameters()),
        "act": _act_str,
        "lr": _lr,
        "h": _h,
        "epochs": _n_ep,
    }
    _fig
    return (xor_results,)


@app.cell
def _(mo, xor_results):
    _p = xor_results["preds"]
    _converged = all(abs(_p[_j] - t) < 0.1 for _j, t in enumerate([0, 1, 1, 0]))
    mo.md(
        f"""
        ### Training Results

        | Input | Target | Prediction | Error |
        |-------|--------|------------|-------|
        | (0, 0) | 0 | {_p[0]:.4f} | {abs(_p[0]):.4f} |
        | (0, 1) | 1 | {_p[1]:.4f} | {abs(_p[1] - 1):.4f} |
        | (1, 0) | 1 | {_p[2]:.4f} | {abs(_p[2] - 1):.4f} |
        | (1, 1) | 0 | {_p[3]:.4f} | {abs(_p[3]):.4f} |
        | | | **Final MSE** | **{xor_results['final_loss']:.6f}** |

        {'✅ **Converged!** All predictions within 0.1 of target.' if _converged else '⚠️ **Not fully converged.** Try: more epochs, higher learning rate, or more hidden neurons.'}

        **Trainable parameters:** {xor_results['n_params']}

        ---

        ### Things to Try

        1. Set hidden = 2 — can it still learn XOR?
        2. Compare Sigmoid vs Tanh — which converges faster?
        3. Sweep learning rate: too small (0.1) → too large (5.0) — observe oscillation
        4. Change random seed — sometimes training gets stuck in a local minimum
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary: Math ↔ PyTorch

    | Mathematical Step | PyTorch Code |
    |-------------------|-------------|
    | $\vec{z} = W\vec{a} + \vec{b}$ | `nn.Linear(in, out)` |
    | $\vec{a} = \sigma(\vec{z})$ | `nn.Sigmoid()` |
    | Forward pass: $y = f(x; W, b)$ | `y = model(x)` |
    | Loss: $L = \frac{1}{N}\sum\lVert\bar{y}-y\rVert^2$ | `nn.MSELoss()` |
    | Reset gradients to zero | `optimizer.zero_grad()` |
    | Backprop: compute all $\frac{\partial L}{\partial w}$ | `loss.backward()` |
    | Update: $w \leftarrow w - r \nabla_w L$ | `optimizer.step()` |

    **Key insight:** `loss.backward()` is backpropagation (computes the direction).
    `optimizer.step()` is gradient descent (takes the step).
    """)
    return


if __name__ == "__main__":
    app.run()
