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
IME 775: Training Neural Networks — Interactive PyTorch Notebook
=================================================================
An interactive marimo notebook covering activation functions, forward
propagation, MSE loss, backpropagation, and the complete training loop.
All tensor values are editable via sliders and text inputs.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 8 — Training Neural Networks
Topics: Sigmoid, Tanh, Forward Propagation, MSE Loss, Backpropagation, Gradient Descent
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
def _(mo):
    mo.md(r"""
    # IME 775: Training Neural Networks — Chapter 8 PyTorch Lab

    ## Learning Objectives

    1. Compute sigmoid and tanh activations in PyTorch and verify derivative formulas
    2. Build custom MLPs and trace forward propagation with editable weights
    3. Compute MSE loss and understand its gradient
    4. Run backpropagation step-by-step and inspect all gradients
    5. Train an MLP on XOR with interactive hyperparameters

    ---

    | Section | Topic | Key Concept |
    |---------|-------|-------------|
    | 1 | Activation Functions | Sigmoid, Tanh, derivatives |
    | 2 | Forward Propagation | Layer-by-layer with editable weights |
    | 3 | MSE Loss & Gradients | Loss surface, autograd |
    | 4 | Backpropagation | Step-by-step gradient inspection |
    | 5 | Full Training Loop | XOR with live loss curve |
    | 6 | Vanishing Gradients | Depth vs gradient magnitude |
    """)
    return


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
    ---
    ## 1. Activation Functions: Sigmoid & Tanh

    $$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)$$

    **Edit the input values below** to compute activations and derivatives interactively.
    """)
    return


@app.cell
def _(mo):
    act_x1 = mo.ui.number(value=-3.0, start=-10.0, stop=10.0, step=0.1,
                           label="x₁")
    act_x2 = mo.ui.number(value=-1.0, start=-10.0, stop=10.0, step=0.1,
                           label="x₂")
    act_x3 = mo.ui.number(value=0.0, start=-10.0, stop=10.0, step=0.1,
                           label="x₃")
    act_x4 = mo.ui.number(value=1.0, start=-10.0, stop=10.0, step=0.1,
                           label="x₄")
    act_x5 = mo.ui.number(value=3.0, start=-10.0, stop=10.0, step=0.1,
                           label="x₅")
    mo.md(
        f"""
        ### Edit Input Tensor Values

        | x₁ | x₂ | x₃ | x₄ | x₅ |
        |----|----|----|----|----|
        | {act_x1} | {act_x2} | {act_x3} | {act_x4} | {act_x5} |
        """
    )
    return act_x1, act_x2, act_x3, act_x4, act_x5


@app.cell
def _(act_x1, act_x2, act_x3, act_x4, act_x5, np, plt, torch):
    _vals = [act_x1.value, act_x2.value, act_x3.value, act_x4.value, act_x5.value]
    _x = torch.tensor(_vals, dtype=torch.float32)

    _sig = torch.sigmoid(_x)
    _sig_deriv = _sig * (1 - _sig)
    _tanh_out = torch.tanh(_x)
    _tanh_deriv = 1 - _tanh_out ** 2

    _xc = np.linspace(-6, 6, 300)
    _sig_c = 1 / (1 + np.exp(-_xc))
    _tanh_c = np.tanh(_xc)
    _sig_d_c = _sig_c * (1 - _sig_c)
    _tanh_d_c = 1 - _tanh_c ** 2

    _fig, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    _ax1.plot(_xc, _sig_c, 'b-', alpha=0.3, linewidth=2)
    _ax1.scatter(_vals, _sig.numpy(), c='#3b82f6', s=80, zorder=5, edgecolors='white', linewidth=1.5)
    for _i, (_xi, _yi) in enumerate(zip(_vals, _sig.numpy())):
        _ax1.annotate(f'{_yi:.4f}', (_xi, _yi), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#3b82f6')
    _ax1.set_title('σ(x) — Sigmoid', fontweight='bold')
    _ax1.axhline(0.5, color='gray', ls='--', alpha=0.3)
    _ax1.axvline(0, color='gray', ls='--', alpha=0.3)
    _ax1.set_ylim(-0.1, 1.1)
    _ax1.set_xlabel('x')
    _ax1.grid(alpha=0.2)

    _ax2.plot(_xc, _sig_d_c, 'b-', alpha=0.3, linewidth=2)
    _ax2.scatter(_vals, _sig_deriv.numpy(), c='#8b5cf6', s=80, zorder=5, edgecolors='white', linewidth=1.5)
    for _i, (_xi, _yi) in enumerate(zip(_vals, _sig_deriv.numpy())):
        _ax2.annotate(f'{_yi:.4f}', (_xi, _yi), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#8b5cf6')
    _ax2.set_title("σ'(x) = σ(x)(1−σ(x))", fontweight='bold')
    _ax2.axhline(0.25, color='red', ls='--', alpha=0.4, label='max = 0.25')
    _ax2.legend(fontsize=9)
    _ax2.set_ylim(-0.05, 0.35)
    _ax2.set_xlabel('x')
    _ax2.grid(alpha=0.2)

    _ax3.plot(_xc, _tanh_c, 'g-', alpha=0.3, linewidth=2)
    _ax3.scatter(_vals, _tanh_out.numpy(), c='#22c55e', s=80, zorder=5, edgecolors='white', linewidth=1.5)
    for _i, (_xi, _yi) in enumerate(zip(_vals, _tanh_out.numpy())):
        _ax3.annotate(f'{_yi:.4f}', (_xi, _yi), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#22c55e')
    _ax3.set_title('tanh(x)', fontweight='bold')
    _ax3.axhline(0, color='gray', ls='--', alpha=0.3)
    _ax3.axvline(0, color='gray', ls='--', alpha=0.3)
    _ax3.set_ylim(-1.3, 1.3)
    _ax3.set_xlabel('x')
    _ax3.grid(alpha=0.2)

    _ax4.plot(_xc, _tanh_d_c, 'g-', alpha=0.3, linewidth=2)
    _ax4.scatter(_vals, _tanh_deriv.numpy(), c='#f59e0b', s=80, zorder=5, edgecolors='white', linewidth=1.5)
    for _i, (_xi, _yi) in enumerate(zip(_vals, _tanh_deriv.numpy())):
        _ax4.annotate(f'{_yi:.4f}', (_xi, _yi), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#f59e0b')
    _ax4.set_title("tanh'(x) = 1 − tanh²(x)", fontweight='bold')
    _ax4.axhline(1.0, color='red', ls='--', alpha=0.4, label='max = 1.0')
    _ax4.legend(fontsize=9)
    _ax4.set_ylim(-0.1, 1.2)
    _ax4.set_xlabel('x')
    _ax4.grid(alpha=0.2)

    _fig.suptitle("Activation Functions & Derivatives (editable inputs)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    act_results = {
        "vals": _vals,
        "sig": _sig.numpy().tolist(),
        "sig_d": _sig_deriv.numpy().tolist(),
        "tanh": _tanh_out.numpy().tolist(),
        "tanh_d": _tanh_deriv.numpy().tolist(),
    }
    _fig
    return (act_results,)


@app.cell
def _(act_results, mo):
    _v = act_results["vals"]
    _s = act_results["sig"]
    _sd = act_results["sig_d"]
    _t = act_results["tanh"]
    _td = act_results["tanh_d"]
    mo.md(
        f"""
        ### Results Table

        | x | σ(x) | σ'(x) | tanh(x) | tanh'(x) |
        |---|------|-------|---------|----------|
        | {_v[0]:.1f} | {_s[0]:.4f} | {_sd[0]:.4f} | {_t[0]:.4f} | {_td[0]:.4f} |
        | {_v[1]:.1f} | {_s[1]:.4f} | {_sd[1]:.4f} | {_t[1]:.4f} | {_td[1]:.4f} |
        | {_v[2]:.1f} | {_s[2]:.4f} | {_sd[2]:.4f} | {_t[2]:.4f} | {_td[2]:.4f} |
        | {_v[3]:.1f} | {_s[3]:.4f} | {_sd[3]:.4f} | {_t[3]:.4f} | {_td[3]:.4f} |
        | {_v[4]:.1f} | {_s[4]:.4f} | {_sd[4]:.4f} | {_t[4]:.4f} | {_td[4]:.4f} |

        **Ratio at x=0:** tanh'(0) / σ'(0) = 1.0 / 0.25 = **4×** — tanh gradients are 4 times stronger!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Forward Propagation with Editable Weights

    Build a **2 → 2 → 1** MLP with sigmoid activations and trace the forward pass.
    Edit the weights, biases, and input values to see how the output changes.

    $$\vec{z}^{(l)} = W^{(l)} \vec{a}^{(l-1)} + \vec{b}^{(l)}, \quad \vec{a}^{(l)} = \sigma(\vec{z}^{(l)})$$
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Input & Layer 0 Weights (2 → 2)
    """)
    return


@app.cell
def _(mo):
    fp_x1 = mo.ui.number(value=1.0, start=-5.0, stop=5.0, step=0.1, label="x₁")
    fp_x2 = mo.ui.number(value=0.5, start=-5.0, stop=5.0, step=0.1, label="x₂")

    fp_w00 = mo.ui.number(value=0.5, start=-3.0, stop=3.0, step=0.1, label="w₀₀")
    fp_w01 = mo.ui.number(value=0.4, start=-3.0, stop=3.0, step=0.1, label="w₀₁")
    fp_w10 = mo.ui.number(value=0.3, start=-3.0, stop=3.0, step=0.1, label="w₁₀")
    fp_w11 = mo.ui.number(value=-0.2, start=-3.0, stop=3.0, step=0.1, label="w₁₁")
    fp_b00 = mo.ui.number(value=-0.1, start=-3.0, stop=3.0, step=0.1, label="b₀")
    fp_b01 = mo.ui.number(value=0.2, start=-3.0, stop=3.0, step=0.1, label="b₁")
    mo.md(
        f"""
        | Input | Value | | Layer 0 Weights | Value |
        |-------|-------|-|-----------------|-------|
        | x₁ | {fp_x1} | | w₀₀ | {fp_w00} |
        | x₂ | {fp_x2} | | w₀₁ | {fp_w01} |
        | | | | w₁₀ | {fp_w10} |
        | | | | w₁₁ | {fp_w11} |
        | | | | b₀ | {fp_b00} |
        | | | | b₁ | {fp_b01} |
        """
    )
    return fp_b00, fp_b01, fp_w00, fp_w01, fp_w10, fp_w11, fp_x1, fp_x2


@app.cell
def _(mo):
    mo.md("""
    ### Layer 1 Weights (2 → 1)
    """)
    return


@app.cell
def _(mo):
    fp_v0 = mo.ui.number(value=0.6, start=-3.0, stop=3.0, step=0.1, label="v₀")
    fp_v1 = mo.ui.number(value=-0.5, start=-3.0, stop=3.0, step=0.1, label="v₁")
    fp_bv = mo.ui.number(value=0.1, start=-3.0, stop=3.0, step=0.1, label="b_out")
    mo.md(
        f"""
        | Layer 1 Weight | Value |
        |----------------|-------|
        | v₀ | {fp_v0} |
        | v₁ | {fp_v1} |
        | b_out | {fp_bv} |
        """
    )
    return fp_bv, fp_v0, fp_v1


@app.cell
def _(
    fp_b00,
    fp_b01,
    fp_bv,
    fp_v0,
    fp_v1,
    fp_w00,
    fp_w01,
    fp_w10,
    fp_w11,
    fp_x1,
    fp_x2,
    mo,
    nn,
    torch,
):
    _model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Sigmoid(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    with torch.no_grad():
        _model[0].weight.copy_(torch.tensor([[fp_w00.value, fp_w01.value],
                                             [fp_w10.value, fp_w11.value]]))
        _model[0].bias.copy_(torch.tensor([fp_b00.value, fp_b01.value]))
        _model[2].weight.copy_(torch.tensor([[fp_v0.value, fp_v1.value]]))
        _model[2].bias.copy_(torch.tensor([fp_bv.value]))

    _x = torch.tensor([[fp_x1.value, fp_x2.value]])

    _z0 = _model[0](_x)
    _a0 = _model[1](_z0)
    _z1 = _model[2](_a0)
    _y = _model[3](_z1)

    mo.md(
        f"""
        ### Forward Pass Trace

        **Layer 0 — Pre-activation:**
        $$\\vec{{z}}^{{(0)}} = W^{{(0)}} \\vec{{x}} + \\vec{{b}}^{{(0)}}$$

        | | Computation | Value |
        |---|-----------|-------|
        | z₀⁰ | {fp_w00.value}×{fp_x1.value} + {fp_w01.value}×{fp_x2.value} + ({fp_b00.value}) | **{_z0[0,0].item():.4f}** |
        | z₁⁰ | {fp_w10.value}×{fp_x1.value} + {fp_w11.value}×{fp_x2.value} + ({fp_b01.value}) | **{_z0[0,1].item():.4f}** |

        **Layer 0 — Activation:**
        $$\\vec{{a}}^{{(0)}} = \\sigma(\\vec{{z}}^{{(0)}})$$

        | | Value |
        |---|-------|
        | a₀⁰ = σ({_z0[0,0].item():.4f}) | **{_a0[0,0].item():.4f}** |
        | a₁⁰ = σ({_z0[0,1].item():.4f}) | **{_a0[0,1].item():.4f}** |

        **Layer 1 — Pre-activation & Output:**
        $$z^{{(1)}} = \\vec{{v}}^T \\vec{{a}}^{{(0)}} + b_{{out}}$$

        | | Computation | Value |
        |---|-----------|-------|
        | z¹ | {fp_v0.value}×{_a0[0,0].item():.4f} + {fp_v1.value}×{_a0[0,1].item():.4f} + {fp_bv.value} | **{_z1[0,0].item():.4f}** |
        | **y** | σ({_z1[0,0].item():.4f}) | **{_y[0,0].item():.4f}** |

        PyTorch `model(x)` output: **{_y[0,0].item():.6f}** ✓
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. MSE Loss & Autograd

    $$L = \frac{1}{2} \|\bar{y} - y\|^2$$

    Edit target value and predictions to see how MSE loss and its gradient change.
    PyTorch's autograd computes $\frac{\partial L}{\partial y}$ automatically.
    """)
    return


@app.cell
def _(mo):
    loss_target = mo.ui.number(value=1.0, start=-2.0, stop=2.0, step=0.1,
                                label="Target ȳ")
    loss_pred1 = mo.ui.number(value=0.8, start=-2.0, stop=2.0, step=0.1,
                               label="Pred y₁")
    loss_pred2 = mo.ui.number(value=0.3, start=-2.0, stop=2.0, step=0.1,
                               label="Pred y₂")
    loss_pred3 = mo.ui.number(value=0.6, start=-2.0, stop=2.0, step=0.1,
                               label="Pred y₃")
    mo.md(
        f"""
        ### Editable Predictions vs Target

        | Target | Prediction 1 | Prediction 2 | Prediction 3 |
        |--------|-------------|-------------|-------------|
        | {loss_target} | {loss_pred1} | {loss_pred2} | {loss_pred3} |
        """
    )
    return loss_pred1, loss_pred2, loss_pred3, loss_target


@app.cell
def _(loss_pred1, loss_pred2, loss_pred3, loss_target, np, plt, torch):
    _t = loss_target.value
    _preds = [loss_pred1.value, loss_pred2.value, loss_pred3.value]

    _y = torch.tensor(_preds, requires_grad=True)
    _y_bar = torch.tensor([_t, _t, _t])

    _loss_per = 0.5 * (_y_bar - _y) ** 2
    _total_loss = _loss_per.sum()
    _total_loss.backward()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _colors = ['#3b82f6', '#8b5cf6', '#f59e0b']
    _bars = _ax1.bar(['y₁', 'y₂', 'y₃'], _loss_per.detach().numpy(), color=_colors,
                      edgecolor='white', linewidth=1.5)
    for _bar, _l in zip(_bars, _loss_per.detach().numpy()):
        _ax1.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.005,
                 f'{_l:.4f}', ha='center', va='bottom', fontweight='bold')
    _ax1.set_title(f'Per-Example Loss ℓᵢ = ½(ȳ − yᵢ)²', fontweight='bold')
    _ax1.set_ylabel('Loss')
    _ax1.axhline(0, color='gray', alpha=0.3)

    _yc = np.linspace(_t - 1.5, _t + 1.5, 200)
    _lc = 0.5 * (_t - _yc) ** 2
    _ax2.plot(_yc, _lc, 'b-', alpha=0.4, linewidth=2, label='L(y) = ½(ȳ−y)²')
    for _i, (_p, _g) in enumerate(zip(_preds, _y.grad.numpy())):
        _l_val = 0.5 * (_t - _p) ** 2
        _ax2.scatter([_p], [_l_val], c=_colors[_i], s=100, zorder=5,
                     edgecolors='white', linewidth=1.5)
        _arrow_len = min(0.8, abs(_g) * 0.5)
        _dir = -1 if _g > 0 else 1
        _ax2.annotate('', xy=(_p + _dir * _arrow_len, _l_val),
                      xytext=(_p, _l_val),
                      arrowprops=dict(arrowstyle='->', color=_colors[_i], lw=2))
        _ax2.annotate(f'∂L/∂y={_g:.3f}', (_p, _l_val + 0.03),
                     fontsize=8, ha='center', color=_colors[_i], fontweight='bold')
    _ax2.axvline(_t, color='red', ls='--', alpha=0.5, label=f'target ȳ={_t}')
    _ax2.set_title('Loss Curve & Gradients', fontweight='bold')
    _ax2.set_xlabel('y (prediction)')
    _ax2.set_ylabel('Loss')
    _ax2.legend(fontsize=9)
    _ax2.grid(alpha=0.2)

    _fig.suptitle("MSE Loss with PyTorch Autograd", fontsize=14, fontweight='bold')
    plt.tight_layout()

    loss_results = {
        "preds": _preds,
        "loss_per": _loss_per.detach().numpy().tolist(),
        "total": _total_loss.item(),
        "grads": _y.grad.numpy().tolist(),
        "target": _t,
    }
    _fig
    return (loss_results,)


@app.cell
def _(loss_results, mo):
    _p = loss_results["preds"]
    _lp = loss_results["loss_per"]
    _g = loss_results["grads"]
    _t = loss_results["target"]
    mo.md(
        f"""
        ### Autograd Results

        | Prediction | Loss ℓᵢ | ∂L/∂yᵢ = −(ȳ−yᵢ) | Direction |
        |-----------|---------|-------------------|-----------|
        | y₁ = {_p[0]:.1f} | {_lp[0]:.4f} | {_g[0]:.4f} | {'← decrease y' if _g[0] > 0 else '→ increase y'} |
        | y₂ = {_p[1]:.1f} | {_lp[1]:.4f} | {_g[1]:.4f} | {'← decrease y' if _g[1] > 0 else '→ increase y'} |
        | y₃ = {_p[2]:.1f} | {_lp[2]:.4f} | {_g[2]:.4f} | {'← decrease y' if _g[2] > 0 else '→ increase y'} |
        | **Total** | **{loss_results['total']:.4f}** | — | — |

        The gradient always points **away** from the target. Gradient descent (subtracting the gradient)
        moves the prediction **toward** the target. ✓
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Backpropagation Step-by-Step

    We build a **1 → 2 → 1** MLP, run a forward pass, then use PyTorch autograd
    to compute all gradients. Compare with the hand-computed backpropagation formulas.

    Edit the weights and input to see how gradients change.
    """)
    return


@app.cell
def _(mo):
    bp_x = mo.ui.number(value=1.0, start=-3.0, stop=3.0, step=0.1, label="Input x")
    bp_target = mo.ui.number(value=0.0, start=-1.0, stop=2.0, step=0.1, label="Target ȳ")
    bp_w00 = mo.ui.number(value=0.5, start=-3.0, stop=3.0, step=0.1, label="w₀₀")
    bp_w10 = mo.ui.number(value=-0.3, start=-3.0, stop=3.0, step=0.1, label="w₁₀")
    bp_b00 = mo.ui.number(value=0.1, start=-3.0, stop=3.0, step=0.1, label="b₀₀")
    bp_b10 = mo.ui.number(value=-0.1, start=-3.0, stop=3.0, step=0.1, label="b₁₀")
    bp_v0 = mo.ui.number(value=0.4, start=-3.0, stop=3.0, step=0.1, label="v₀")
    bp_v1 = mo.ui.number(value=0.6, start=-3.0, stop=3.0, step=0.1, label="v₁")
    bp_bv = mo.ui.number(value=0.2, start=-3.0, stop=3.0, step=0.1, label="b_out")
    bp_lr = mo.ui.slider(start=0.01, stop=5.0, step=0.01, value=1.0,
                          label="Learning rate r", show_value=True)
    mo.md(
        f"""
        ### Network Configuration (1 → 2 → 1)

        | Layer 0 | Value | | Layer 1 | Value | | Training | Value |
        |---------|-------|-|---------|-------|-|----------|-------|
        | w₀₀ | {bp_w00} | | v₀ | {bp_v0} | | x | {bp_x} |
        | w₁₀ | {bp_w10} | | v₁ | {bp_v1} | | ȳ | {bp_target} |
        | b₀₀ | {bp_b00} | | b_out | {bp_bv} | | r | {bp_lr} |
        | b₁₀ | {bp_b10} | | | | | | |
        """
    )
    return (
        bp_b00,
        bp_b10,
        bp_bv,
        bp_lr,
        bp_target,
        bp_v0,
        bp_v1,
        bp_w00,
        bp_w10,
        bp_x,
    )


@app.cell
def _(
    bp_b00,
    bp_b10,
    bp_bv,
    bp_lr,
    bp_target,
    bp_v0,
    bp_v1,
    bp_w00,
    bp_w10,
    bp_x,
    mo,
    nn,
    torch,
):
    _model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        _model[0].weight.copy_(torch.tensor([[bp_w00.value], [bp_w10.value]]))
        _model[0].bias.copy_(torch.tensor([bp_b00.value, bp_b10.value]))
        _model[2].weight.copy_(torch.tensor([[bp_v0.value, bp_v1.value]]))
        _model[2].bias.copy_(torch.tensor([bp_bv.value]))

    _x = torch.tensor([[bp_x.value]])
    _y_bar = torch.tensor([[bp_target.value]])

    _z0 = _model[0](_x)
    _a0 = _model[1](_z0)
    _z1 = _model[2](_a0)
    _y = _model[3](_z1)

    _loss = 0.5 * (_y_bar - _y) ** 2
    _loss.backward()

    _gw0 = _model[0].weight.grad.clone()
    _gb0 = _model[0].bias.grad.clone()
    _gw1 = _model[2].weight.grad.clone()
    _gb1 = _model[2].bias.grad.clone()

    _d1 = -(_y_bar.item() - _y.item()) * _y.item() * (1 - _y.item())
    _sig0 = _a0.detach()
    _d00 = _d1 * bp_v0.value * _sig0[0, 0].item() * (1 - _sig0[0, 0].item())
    _d10 = _d1 * bp_v1.value * _sig0[0, 1].item() * (1 - _sig0[0, 1].item())

    _r = bp_lr.value
    _w00_new = bp_w00.value - _r * _gw0[0, 0].item()
    _w10_new = bp_w10.value - _r * _gw0[1, 0].item()
    _b00_new = bp_b00.value - _r * _gb0[0].item()
    _b10_new = bp_b10.value - _r * _gb0[1].item()
    _v0_new = bp_v0.value - _r * _gw1[0, 0].item()
    _v1_new = bp_v1.value - _r * _gw1[0, 1].item()
    _bv_new = bp_bv.value - _r * _gb1[0].item()

    _model2 = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        _model2[0].weight.copy_(torch.tensor([[_w00_new], [_w10_new]]))
        _model2[0].bias.copy_(torch.tensor([_b00_new, _b10_new]))
        _model2[2].weight.copy_(torch.tensor([[_v0_new, _v1_new]]))
        _model2[2].bias.copy_(torch.tensor([_bv_new]))
    _y_new = _model2(_x).item()
    _loss_new = 0.5 * (_y_bar.item() - _y_new) ** 2

    mo.md(
        f"""
        ### Forward Pass

        | Variable | Formula | Value |
        |----------|---------|-------|
        | z₀⁰ | w₀₀·x + b₀₀ | **{_z0[0,0].item():.4f}** |
        | z₁⁰ | w₁₀·x + b₁₀ | **{_z0[0,1].item():.4f}** |
        | a₀⁰ | σ(z₀⁰) | **{_a0[0,0].item():.4f}** |
        | a₁⁰ | σ(z₁⁰) | **{_a0[0,1].item():.4f}** |
        | z¹ | v₀·a₀ + v₁·a₁ + b | **{_z1[0,0].item():.4f}** |
        | **y** | σ(z¹) | **{_y.item():.4f}** |
        | **Loss** | ½(ȳ − y)² | **{_loss.item():.6f}** |

        ### Backward Pass (Backpropagation)

        | Variable | Manual δ | PyTorch ∂L/∂ | Match? |
        |----------|----------|-------------|--------|
        | δ¹ | {_d1:.6f} | — | — |
        | ∂L/∂v₀ | δ¹·a₀ = {_d1*_sig0[0,0].item():.6f} | {_gw1[0,0].item():.6f} | {'✓' if abs(_d1*_sig0[0,0].item() - _gw1[0,0].item()) < 1e-4 else '✗'} |
        | ∂L/∂v₁ | δ¹·a₁ = {_d1*_sig0[0,1].item():.6f} | {_gw1[0,1].item():.6f} | {'✓' if abs(_d1*_sig0[0,1].item() - _gw1[0,1].item()) < 1e-4 else '✗'} |
        | ∂L/∂b_out | δ¹ = {_d1:.6f} | {_gb1[0].item():.6f} | {'✓' if abs(_d1 - _gb1[0].item()) < 1e-4 else '✗'} |
        | δ₀₀ | {_d00:.6f} | — | — |
        | δ₁₀ | {_d10:.6f} | — | — |
        | ∂L/∂w₀₀ | δ₀₀·x = {_d00*bp_x.value:.6f} | {_gw0[0,0].item():.6f} | {'✓' if abs(_d00*bp_x.value - _gw0[0,0].item()) < 1e-4 else '✗'} |
        | ∂L/∂w₁₀ | δ₁₀·x = {_d10*bp_x.value:.6f} | {_gw0[1,0].item():.6f} | {'✓' if abs(_d10*bp_x.value - _gw0[1,0].item()) < 1e-4 else '✗'} |

        ### Weight Update (r = {_r})

        | Param | Old | − r·grad | New |
        |-------|-----|---------|-----|
        | w₀₀ | {bp_w00.value:.4f} | {-_r*_gw0[0,0].item():+.6f} | {_w00_new:.4f} |
        | w₁₀ | {bp_w10.value:.4f} | {-_r*_gw0[1,0].item():+.6f} | {_w10_new:.4f} |
        | v₀ | {bp_v0.value:.4f} | {-_r*_gw1[0,0].item():+.6f} | {_v0_new:.4f} |
        | v₁ | {bp_v1.value:.4f} | {-_r*_gw1[0,1].item():+.6f} | {_v1_new:.4f} |

        ### Did it improve?

        | | Before | After |
        |---|--------|-------|
        | y | {_y.item():.4f} | {_y_new:.4f} |
        | Loss | {_loss.item():.6f} | {_loss_new:.6f} |
        | | | {'✅ Loss decreased!' if _loss_new < _loss.item() else '⚠️ Loss increased (lr too high?)'} |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Full Training Loop: Solving XOR

    Train a neural network on the XOR problem with interactive hyperparameters.
    **Adjust the settings, then click "Train" to run.**

    | Input | Target |
    |-------|--------|
    | (0,0) | 0 |
    | (0,1) | 1 |
    | (1,0) | 1 |
    | (1,1) | 0 |
    """)
    return


@app.cell
def _(mo):

    xor_form = (
        mo.md(
            """
            **Learning rate:** {lr}

            **Hidden neurons per layer:** {hidden}

            **Epochs:** {epochs}

            **Activation:** {activation}

            **Random seed:** {seed}
            """
        )
        .batch(
            lr=mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=2.0,
                            label="Learning rate"),
            hidden=mo.ui.slider(start=2, stop=16, step=1, value=4,
                                label="Hidden neurons per layer"),
            epochs=mo.ui.slider(start=1000, stop=20000, step=1000, value=5000,
                                label="Epochs"),
            activation=mo.ui.dropdown(options={"Sigmoid": "Sigmoid", "Tanh": "Tanh"},
                                       value="Sigmoid", label="Activation"),
            seed=mo.ui.number(value=42, start=0, stop=999, step=1,
                              label="Random seed"),
        )
        .form(submit_button_label="🚀 Train Network")
    )
    xor_form



    return (xor_form,)


@app.cell
def _(mo, nn, np, plt, torch, xor_form):
    mo.stop(xor_form.value is None, mo.md("*Adjust hyperparameters above and click **Train Network** to start.*"))

    _cfg = xor_form.value
    _lr_val = _cfg["lr"]
    _h = int(_cfg["hidden"])
    _n_epochs = int(_cfg["epochs"])
    _act_choice = _cfg["activation"]
    _seed = int(_cfg["seed"])
    _act_name = "Sigmoid" if _act_choice == "sigmoid" else "Tanh"

    torch.manual_seed(_seed)

    def _make_act():
        return nn.Sigmoid() if _act_choice == "sigmoid" else nn.Tanh()

    _model = nn.Sequential(
        nn.Linear(2, _h),
        _make_act(),
        nn.Linear(_h, _h),
        _make_act(),
        nn.Linear(_h, 1),
        nn.Sigmoid()
    )

    _X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    _Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    _loss_fn = nn.MSELoss()
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_lr_val)

    _losses = []
    for _epoch in range(_n_epochs):
        _y_pred = _model(_X)
        _loss = _loss_fn(_y_pred, _Y)
        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
        if _epoch % max(1, _n_epochs // 500) == 0 or _epoch == _n_epochs - 1:
            _losses.append((_epoch, _loss.item()))

    with torch.no_grad():
        _final_pred = _model(_X)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _ep = [e for e, _ in _losses]
    _lv = [v for _, v in _losses]
    _ax1.plot(_ep, _lv, 'b-', linewidth=1.5)
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('MSE Loss')
    _ax1.set_title(f'Training Loss ({_act_name}, lr={_lr_val}, h={_h})', fontweight='bold')
    _min_loss = min(_lv)
    if _min_loss > 0:
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
    _colors_xor = ['#3b82f6' if y == 0 else '#ef4444' for y in _Y.numpy().flatten()]
    _ax2.scatter(_X[:, 0].numpy(), _X[:, 1].numpy(), c=_colors_xor, s=200,
                 edgecolors='white', linewidth=2, zorder=5)
    for _i in range(4):
        _ax2.annotate(f'{_final_pred[_i].item():.3f}',
                     (_X[_i, 0].item(), _X[_i, 1].item()),
                     textcoords="offset points", xytext=(15, -5),
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    _ax2.set_title('Decision Boundary', fontweight='bold')
    _ax2.set_xlabel('x₁')
    _ax2.set_ylabel('x₂')

    _fig.suptitle(f"XOR Training: {_act_name} | {_n_epochs} epochs", fontsize=14, fontweight='bold')
    plt.tight_layout()

    xor_results = {
        "preds": [_final_pred[i].item() for i in range(4)],
        "final_loss": _losses[-1][1],
        "act_name": _act_name,
        "lr": _lr_val,
        "hidden": _h,
        "epochs": _n_epochs,
        "n_params": sum(p.numel() for p in _model.parameters()),
    }
    _fig
    return (xor_results,)


@app.cell
def _(mo, xor_results):
    _p = xor_results["preds"]
    _converged = all(abs(_p[i] - t) < 0.1 for i, t in enumerate([0, 1, 1, 0]))
    mo.md(
        f"""
        ### Training Results ({xor_results['act_name']}, lr={xor_results['lr']}, h={xor_results['hidden']}, {xor_results['epochs']} epochs)

        | Input | Target | Prediction | Error |
        |-------|--------|------------|-------|
        | (0, 0) | 0 | {_p[0]:.4f} | {abs(_p[0]):.4f} |
        | (0, 1) | 1 | {_p[1]:.4f} | {abs(_p[1] - 1):.4f} |
        | (1, 0) | 1 | {_p[2]:.4f} | {abs(_p[2] - 1):.4f} |
        | (1, 1) | 0 | {_p[3]:.4f} | {abs(_p[3]):.4f} |
        | | | **Final Loss** | **{xor_results['final_loss']:.6f}** |

        {'✅ **Converged!** All predictions within 0.1 of target.' if _converged else '⚠️ **Not fully converged.** Try: more epochs, higher learning rate, or more hidden neurons.'}

        **Parameters trained:** {xor_results['n_params']}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Vanishing Gradients: Depth vs Gradient Magnitude

    Sigmoid's maximum derivative is 0.25. After $L$ layers, the gradient at the first layer
    shrinks by up to $0.25^L$. This demo shows gradient magnitudes across layers for networks of varying depth.
    """)
    return


@app.cell
def _(mo):
    vg_depth = mo.ui.slider(start=2, stop=20, step=1, value=5,
                             label="Network depth (layers)", show_value=True)
    vg_width = mo.ui.slider(start=2, stop=16, step=1, value=4,
                             label="Layer width", show_value=True)
    vg_act = mo.ui.dropdown(options={"Sigmoid": "sigmoid", "Tanh": "tanh"},
                             value="sigmoid", label="Activation")
    mo.md(
        f"""
        | Control | Setting |
        |---------|---------|
        | Depth | {vg_depth} |
        | Width | {vg_width} |
        | Activation | {vg_act} |
        """
    )
    return vg_act, vg_depth, vg_width


@app.cell
def _(nn, np, plt, torch, vg_act, vg_depth, vg_width):
    torch.manual_seed(0)
    _d = int(vg_depth.value)
    _w = int(vg_width.value)

    _layers = []
    _in = 2
    for _i in range(_d):
        _out = _w if _i < _d - 1 else 1
        _layers.append(nn.Linear(_in, _out))
        if vg_act.value == "sigmoid":
            _layers.append(nn.Sigmoid())
        else:
            _layers.append(nn.Tanh())
        _in = _out

    _model = nn.Sequential(*_layers)

    _x = torch.tensor([[0.5, -0.5]])
    _y_bar = torch.tensor([[1.0]])
    _y_pred = _model(_x)
    _loss = 0.5 * (_y_bar - _y_pred) ** 2
    _loss.backward()

    _grad_norms = []
    _layer_names = []
    _idx = 0
    for _name, _param in _model.named_parameters():
        if 'weight' in _name:
            _norm = _param.grad.norm().item()
            _grad_norms.append(_norm)
            _layer_names.append(f'L{_idx}')
            _idx += 1

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _colors_vg = plt.cm.viridis(np.linspace(0.2, 0.9, len(_grad_norms)))
    _ax1.bar(_layer_names, _grad_norms, color=_colors_vg, edgecolor='white', linewidth=1.5)
    _ax1.set_xlabel('Layer (0 = first hidden)')
    _ax1.set_ylabel('||∂L/∂W|| (gradient norm)')
    _ax1.set_title(f'Gradient Norms ({vg_act.value}, {_d} layers)', fontweight='bold')
    _ax1.grid(alpha=0.2, axis='y')

    _ax2.bar(_layer_names, _grad_norms, color=_colors_vg, edgecolor='white', linewidth=1.5)
    _ax2.set_xlabel('Layer (0 = first hidden)')
    _ax2.set_ylabel('||∂L/∂W|| (log scale)')
    _ax2.set_yscale('log')
    _ax2.set_title(f'Gradient Norms — Log Scale', fontweight='bold')
    _ax2.grid(alpha=0.2, axis='y')

    _fig.suptitle("Vanishing Gradient Problem", fontsize=14, fontweight='bold')
    plt.tight_layout()

    vg_results = {
        "grad_norms": _grad_norms,
        "depth": _d,
        "act": vg_act.value,
    }
    _fig
    return (vg_results,)


@app.cell
def _(mo, vg_results):
    _gn = vg_results["grad_norms"]
    _d = vg_results["depth"]
    _act = vg_results["act"]
    _ratio = _gn[0] / _gn[-1] if _gn[-1] > 1e-30 else float('inf')
    mo.md(
        f"""
        ### Analysis

        | Metric | Value |
        |--------|-------|
        | First layer gradient norm | {_gn[0]:.2e} |
        | Last layer gradient norm | {_gn[-1]:.2e} |
        | Ratio (first / last) | {_ratio:.2e} |
        | Theoretical max shrink per layer | {'0.25 (sigmoid)' if _act == 'sigmoid' else '1.0 (tanh)'} |
        | Expected ratio for {_d} layers | {0.25**(_d-1):.2e if _act == 'sigmoid' else '~1'} |

        {'⚠️ **Vanishing gradients detected!** The first layer gradient is orders of magnitude smaller than the last.' if _ratio < 0.1 else '✅ Gradients are reasonably balanced across layers.'}

        {'Try switching to **Tanh** or reducing depth to see the difference.' if _act == 'sigmoid' and _d > 4 else ''}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary: PyTorch ↔ Math Mapping

    | Mathematical Concept | PyTorch Code |
    |---------------------|-------------|
    | $\sigma(x) = \frac{1}{1+e^{-x}}$ | `torch.sigmoid(x)` |
    | $\tanh(x)$ | `torch.tanh(x)` |
    | $\vec{z} = W\vec{a} + \vec{b}$ | `nn.Linear(in, out)` |
    | $\vec{a} = \sigma(\vec{z})$ | `nn.Sigmoid()` |
    | Forward pass | `y = model(x)` |
    | $L = \frac{1}{N}\sum\|\bar{y}-y\|^2$ | `nn.MSELoss()` |
    | Reset $\nabla \to 0$ | `optimizer.zero_grad()` |
    | Backprop: compute all $\frac{\partial L}{\partial w}$ | `loss.backward()` |
    | $w \leftarrow w - r \nabla_w L$ | `optimizer.step()` |

    ### Key Insights from This Lab

    1. **Sigmoid derivative** $\sigma'(x) = \sigma(x)(1-\sigma(x))$ — max 0.25 at $x=0$
    2. **Tanh derivative** $\tanh'(x) = 1-\tanh^2(x)$ — max 1.0 at $x=0$ (4× stronger)
    3. **Autograd** computes the same gradients as manual backpropagation (verified ✓)
    4. **Vanishing gradients** cause first-layer gradients to shrink exponentially with depth
    5. **Learning rate** is critical: too high → divergence, too low → slow convergence
    """)
    return


if __name__ == "__main__":
    app.run()
