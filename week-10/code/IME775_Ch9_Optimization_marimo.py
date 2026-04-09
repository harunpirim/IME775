# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "torch==2.11.0",
# ]
# ///
"""
IME 775: Loss, Optimization & Regularization — Interactive PyTorch Notebook
=============================================================================
An interactive marimo notebook covering loss functions, softmax, optimization
algorithms, and regularization techniques.  All key values are editable.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 9 — Loss, Optimization, and Regularization
Topics: CE loss, Softmax, SGD, Momentum, Adam, L1/L2, Dropout
"""

import marimo

__generated_with = "0.23.0"
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
    # IME 775: Loss, Optimization & Regularization — Chapter 9 PyTorch Lab

    ## Learning Objectives

    1. Compute and compare loss functions (MSE, Cross-Entropy, Focal, Hinge)
    2. Understand softmax as a differentiable approximation to argmax
    3. Visualise and compare optimizers (SGD, Momentum, Nesterov, Adam)
    4. Explore L1 vs L2 regularization and their sparsity properties
    5. Observe the effect of dropout during training

    ---

    | Section | Topic | Key Concept |
    |---------|-------|-------------|
    | 1 | Loss Functions | MSE, CE, Focal, Hinge |
    | 2 | Softmax | Scores → Probabilities |
    | 3 | Optimizer Comparison | SGD → Momentum → Adam |
    | 4 | Regularization | L1 vs L2, weight sparsity |
    | 5 | Dropout | Ensemble of subnetworks |
    | 6 | Full Training: 3-Class Classifier | Putting it all together |
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
    ## 1. Loss Functions

    We compare four loss functions used in neural-network training.

    $$\text{MSE:}\; L = \|{\vec{y}} - \bar{y}\|^2, \qquad
      \text{CE:}\; L = -\sum_j \bar{y}_j \log y_j$$

    **Edit the prediction probabilities below** to see how each loss responds.
    """)
    return


@app.cell
def _(mo):
    loss_p0 = mo.ui.slider(0.01, 0.98, value=0.05, step=0.01, label="P(cat)")
    loss_p1 = mo.ui.slider(0.01, 0.98, value=0.80, step=0.01, label="P(dog)")
    loss_gt = mo.ui.dropdown(
        options={"cat (0)": 0, "dog (1)": 1, "airplane (2)": 2, "auto (3)": 3},
        value="dog (1)",
        label="Ground-truth class",
    )
    mo.md(
        f"""
        ### Edit Prediction & Ground Truth

        | P(cat) | P(dog) | GT class |
        |--------|--------|----------|
        | {loss_p0} | {loss_p1} | {loss_gt} |

        *P(airplane) and P(auto) share the remaining probability equally.*
        """
    )



    return loss_gt, loss_p0, loss_p1


@app.cell
def _(loss_gt, loss_p0, loss_p1, np, plt, torch):
    _p0 = loss_p0.value
    _p1 = loss_p1.value
    _remaining = max(1e-6, 1.0 - _p0 - _p1)
    _probs = torch.tensor([_p0, _p1, _remaining / 2, _remaining / 2],
                           dtype=torch.float32)
    _probs = _probs / _probs.sum()
    _gt_idx = loss_gt.value

    _gt_onehot = torch.zeros(4)
    _gt_onehot[_gt_idx] = 1.0

    _mse = torch.sum((_probs - _gt_onehot) ** 2).item()
    _ce = -torch.log(_probs[_gt_idx]).item()

    _gamma = 2.0
    _pt = _probs[_gt_idx].item()
    _focal = -((1 - _pt) ** _gamma) * np.log(_pt)

    _scores = torch.log(_probs + 1e-8)
    _margin = 1.0
    _hinge = sum(
        max(0, _scores[j].item() - _scores[_gt_idx].item() + _margin)
        for j in range(4) if j != _gt_idx
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))

    _names = ["cat", "dog", "airplane", "auto"]
    _colors = ["#e74c3c" if i == _gt_idx else "#3498db" for i in range(4)]
    _axes[0].bar(_names, _probs.numpy(), color=_colors)
    _axes[0].set_ylabel("Probability")
    _axes[0].set_title(f"Prediction  (GT = {_names[_gt_idx]})")
    _axes[0].set_ylim(0, 1.05)

    _loss_names = ["MSE", "CE", f"Focal(γ={_gamma})", "Hinge(m=1)"]
    _loss_vals = [_mse, _ce, _focal, _hinge]
    _axes[1].barh(_loss_names, _loss_vals, color=["#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"])
    _axes[1].set_xlabel("Loss value")
    _axes[1].set_title("Loss Comparison")

    plt.tight_layout()

    loss_results = {
        "probs": _probs.numpy(),
        "mse": _mse, "ce": _ce, "focal": _focal, "hinge": _hinge,
    }
    _fig
    return (loss_results,)


@app.cell
def _(loss_results, mo):
    _r = loss_results
    mo.md(
        f"""
        | Loss | Value |
        |------|-------|
        | MSE | {_r['mse']:.4f} |
        | Cross-Entropy | {_r['ce']:.4f} |
        | Focal (γ=2) | {_r['focal']:.4f} |
        | Hinge (m=1) | {_r['hinge']:.4f} |

        **Observation:** CE loss depends only on the predicted probability of the
        *correct* class. Focal loss dramatically reduces the contribution of
        easy (high-confidence) examples.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Softmax: Scores → Probabilities

    $$\text{softmax}(\vec{s})_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

    **Edit the raw scores** and observe how softmax converts them to probabilities.
    The **temperature** slider controls sharpness.
    """)
    return


@app.cell
def _(mo):
    sm_s0 = mo.ui.number(value=2.0, start=-20, stop=20, step=0.5, label="s₀")
    sm_s1 = mo.ui.number(value=5.0, start=-20, stop=20, step=0.5, label="s₁")
    sm_s2 = mo.ui.number(value=1.0, start=-20, stop=20, step=0.5, label="s₂")
    sm_s3 = mo.ui.number(value=-1.0, start=-20, stop=20, step=0.5, label="s₃")
    sm_temp = mo.ui.slider(0.1, 10.0, value=1.0, step=0.1, label="Temperature τ")
    mo.md(
        f"""
        ### Edit Score Vector & Temperature

        | s₀ | s₁ | s₂ | s₃ | τ |
        |----|----|----|----|----|
        | {sm_s0} | {sm_s1} | {sm_s2} | {sm_s3} | {sm_temp} |
        """
    )
    return sm_s0, sm_s1, sm_s2, sm_s3, sm_temp


@app.cell
def _(plt, sm_s0, sm_s1, sm_s2, sm_s3, sm_temp, torch):
    _scores = torch.tensor(
        [sm_s0.value, sm_s1.value, sm_s2.value, sm_s3.value],
        dtype=torch.float32,
    )
    _tau = sm_temp.value
    _probs = torch.softmax(_scores / _tau, dim=0)

    _argmax_onehot = torch.zeros(4)
    _argmax_onehot[torch.argmax(_scores)] = 1.0

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.5))

    _names = ["s₀", "s₁", "s₂", "s₃"]

    _axes[0].bar(_names, _scores.numpy(), color="#3498db")
    _axes[0].set_title("Raw Scores")
    _axes[0].axhline(0, color="grey", linewidth=0.5)

    _axes[1].bar(_names, _argmax_onehot.numpy(), color="#e74c3c")
    _axes[1].set_title("Argmax One-Hot")
    _axes[1].set_ylim(0, 1.1)

    _axes[2].bar(_names, _probs.numpy(), color="#2ecc71")
    _axes[2].set_title(f"Softmax (τ = {_tau:.1f})")
    _axes[2].set_ylim(0, 1.1)
    for _i, _p in enumerate(_probs.numpy()):
        _axes[2].text(_i, _p + 0.02, f"{_p:.3f}", ha="center", fontsize=9)

    plt.tight_layout()

    sm_results = {"probs": _probs.numpy(), "scores": _scores.numpy()}
    _fig
    return (sm_results,)


@app.cell
def _(mo, sm_results):
    _p = sm_results["probs"]
    mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        | Sum of probabilities | {_p.sum():.6f} |
        | Max probability | {_p.max():.4f} |
        | Predicted class | {_p.argmax()} |

        **Try:** Set τ → 0.1 (nearly one-hot) vs τ → 10 (nearly uniform).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Optimizer Comparison

    We visualise trajectories of different optimizers on the 2-D loss surface
    $L(w_0, w_1) = w_0^2 + 10\,w_1^2$ (an elongated bowl).

    **Adjust the learning rate** and observe convergence behaviour.
    """)
    return


@app.cell
def _(mo):
    opt_lr = mo.ui.slider(0.005, 0.15, value=0.03, step=0.005, label="Learning rate η")
    opt_steps = mo.ui.slider(10, 200, value=50, step=10, label="Iterations")
    mo.md(
        f"""
        ### Optimizer Settings

        | Learning rate | Iterations |
        |---------------|------------|
        | {opt_lr} | {opt_steps} |
        """
    )
    return opt_lr, opt_steps


@app.cell
def _(np, opt_lr, opt_steps, plt, torch):
    _lr = opt_lr.value
    _n_steps = opt_steps.value

    def _run_optimizer(opt_class, lr, steps, **kwargs):
        w = torch.tensor([4.0, 4.0], requires_grad=True)
        opt = opt_class([w], lr=lr, **kwargs)
        traj = [w.detach().clone().numpy()]
        losses = []
        for _ in range(steps):
            loss = w[0] ** 2 + 10 * w[1] ** 2
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            traj.append(w.detach().clone().numpy())
        return np.array(traj), losses

    _configs = [
        ("SGD", torch.optim.SGD, {"lr": _lr}),
        ("Momentum", torch.optim.SGD, {"lr": _lr, "momentum": 0.9}),
        ("Nesterov", torch.optim.SGD, {"lr": _lr, "momentum": 0.9, "nesterov": True}),
        ("Adam", torch.optim.Adam, {"lr": _lr}),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    _w0 = np.linspace(-5, 5, 100)
    _w1 = np.linspace(-5, 5, 100)
    _W0, _W1 = np.meshgrid(_w0, _w1)
    _Z = _W0 ** 2 + 10 * _W1 ** 2
    _axes[0].contour(_W0, _W1, _Z, levels=20, cmap="coolwarm", alpha=0.6)

    _colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for (_name, _cls, _kw), _c in zip(_configs, _colors):
        _traj, _loss = _run_optimizer(_cls, **_kw, steps=_n_steps)
        _axes[0].plot(_traj[:, 0], _traj[:, 1], "o-", markersize=2,
                      color=_c, label=_name, linewidth=1.2)
        _axes[1].plot(_loss, color=_c, label=_name, linewidth=1.5)

    _axes[0].plot(0, 0, "k*", markersize=12)
    _axes[0].set_xlabel("w₀")
    _axes[0].set_ylabel("w₁")
    _axes[0].set_title("Optimizer Trajectories")
    _axes[0].legend(fontsize=8)
    _axes[0].set_xlim(-5, 5)
    _axes[0].set_ylim(-5, 5)

    _axes[1].set_xlabel("Iteration")
    _axes[1].set_ylabel("Loss")
    _axes[1].set_title("Loss Curves")
    _axes[1].legend(fontsize=8)
    _axes[1].set_yscale("log")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Observations:**
    - **SGD** oscillates along the steep $w_1$ axis while making slow progress on $w_0$.
    - **Momentum** speeds up convergence but may overshoot.
    - **Nesterov** reduces overshooting with look-ahead gradients.
    - **Adam** adapts per-parameter and converges smoothly on both axes.

    Try increasing η to see momentum/SGD diverge while Adam stays stable.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. L1 vs L2 Regularization

    Both add a penalty to the loss to encourage small weights:

    $$L_\text{total} = L(\theta) + \lambda R(\theta)$$

    - **L2:** $R = \|\vec{w}\|^2$ → dense but small weights
    - **L1:** $R = |\vec{w}|$ → sparse weights (many exact zeros)

    **Edit λ** below to see how regularisation changes the optimal weight.
    """)
    return


@app.cell
def _(mo):
    reg_lambda = mo.ui.slider(0.0, 2.0, value=0.3, step=0.05, label="λ (regularization)")
    reg_w_star = mo.ui.number(value=2.0, start=-5.0, stop=5.0, step=0.1,
                               label="True optimum w*")
    mo.md(
        f"""
        ### Settings

        | λ | True w* |
        |---|---------|
        | {reg_lambda} | {reg_w_star} |

        Loss = $(w - w^*)^2$, i.e. the unregularised optimum is at $w = w^*$.
        """
    )
    return reg_lambda, reg_w_star


@app.cell
def _(np, plt, reg_lambda, reg_w_star):
    _lam = reg_lambda.value
    _wstar = reg_w_star.value
    _w = np.linspace(-1, 5, 500)

    _base = (_w - _wstar) ** 2
    _l1 = _base + _lam * np.abs(_w)
    _l2 = _base + _lam * _w ** 2

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].plot(_w, _base, "k--", label="Base loss", linewidth=1)
    _axes[0].plot(_w, _l1, color="#e74c3c", label=f"L1 (λ={_lam:.2f})", linewidth=2)
    _opt_l1 = max(0, _wstar - _lam / 2) if _wstar > 0 else min(0, _wstar + _lam / 2)
    _axes[0].axvline(_opt_l1, color="#e74c3c", linestyle=":", alpha=0.7)
    _axes[0].set_title("L1 Regularization")
    _axes[0].set_xlabel("w")
    _axes[0].set_ylabel("Total loss")
    _axes[0].legend(fontsize=8)
    _axes[0].set_ylim(-0.5, 10)

    _axes[1].plot(_w, _base, "k--", label="Base loss", linewidth=1)
    _axes[1].plot(_w, _l2, color="#3498db", label=f"L2 (λ={_lam:.2f})", linewidth=2)
    _opt_l2 = _wstar / (1 + _lam)
    _axes[1].axvline(_opt_l2, color="#3498db", linestyle=":", alpha=0.7)
    _axes[1].set_title("L2 Regularization")
    _axes[1].set_xlabel("w")
    _axes[1].legend(fontsize=8)
    _axes[1].set_ylim(-0.5, 10)

    plt.tight_layout()

    reg_results = {"opt_l1": float(_opt_l1), "opt_l2": float(_opt_l2)}
    _fig
    return (reg_results,)


@app.cell
def _(mo, reg_results):
    mo.md(f"""
    | | Optimal w |
    |---|---|
    | **No regularization** | w* (as set) |
    | **L1** | {reg_results['opt_l1']:.3f} |
    | **L2** | {reg_results['opt_l2']:.3f} |

    **Key insight:** L1 pushes the optimum all the way to zero for large enough λ.
    L2 shrinks it but never reaches zero. Increase λ to see L1 hit zero first.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Dropout: Simulating Subnetwork Ensembles

    Dropout randomly turns off neurons during training with probability $(1-p)$.
    With $n$ neurons, this simulates $2^n$ subnetworks.

    **Adjust the dropout rate** and observe the variance in layer output.
    """)
    return


@app.cell
def _(mo):
    drop_p = mo.ui.slider(0.0, 0.9, value=0.5, step=0.1, label="Dropout rate (1−p)")
    drop_n_trials = mo.ui.slider(10, 500, value=100, step=10, label="Num forward passes")
    mo.md(
        f"""
        ### Settings

        | Dropout rate | Forward passes |
        |-------------|----------------|
        | {drop_p} | {drop_n_trials} |
        """
    )
    return drop_n_trials, drop_p


@app.cell
def _(drop_n_trials, drop_p, nn, plt, torch):
    _drop_rate = drop_p.value
    _n_trials = int(drop_n_trials.value)

    _layer = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Dropout(p=_drop_rate),
    )
    _layer.train()

    _x = torch.tensor([[1.0, -0.5, 2.0, 0.3]])
    _outputs = []
    for _ in range(_n_trials):
        _outputs.append(_layer(_x).detach().numpy().flatten())

    import numpy as _np
    _outputs = _np.array(_outputs)
    _mean = _outputs.mean(axis=0)
    _std = _outputs.std(axis=0)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _neuron_ids = list(range(8))
    _axes[0].bar(_neuron_ids, _mean, yerr=_std, capsize=3, color="#3498db", alpha=0.8)
    _axes[0].set_xlabel("Neuron index")
    _axes[0].set_ylabel("Output value")
    _axes[0].set_title(f"Mean ± Std over {_n_trials} passes (drop={_drop_rate:.1f})")

    _sums = _outputs.sum(axis=1)
    _axes[1].hist(_sums, bins=30, color="#2ecc71", edgecolor="white", alpha=0.8)
    _axes[1].axvline(_sums.mean(), color="red", linestyle="--", label=f"Mean={_sums.mean():.2f}")
    _axes[1].set_xlabel("Sum of layer output")
    _axes[1].set_ylabel("Frequency")
    _axes[1].set_title("Distribution of Total Activation")
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Try:** Set dropout to 0 (no randomness → all passes identical) vs 0.9
    (most neurons off → high variance). During inference (model.eval()),
    PyTorch disables dropout and uses the full network.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Full Training: 3-Class Classifier with SGD

    We generate a synthetic 2-D dataset (3 clusters) and train a classifier
    using different loss functions and optimizers.

    **Click "Train" after adjusting hyperparameters.**
    """)
    return


@app.cell
def _(mo):
    train_form = mo.md(
        """
        **Learning rate:** {lr}

        **Epochs:** {epochs}

        **Hidden size:** {hidden}

        **Optimizer:** {optimizer}

        **Weight decay (L2):** {weight_decay}

        **Dropout:** {dropout}
        """
    ).batch(
        lr=mo.ui.slider(0.001, 0.5, value=0.05, step=0.005, label="Learning rate"),
        epochs=mo.ui.slider(5, 100, value=30, step=5, label="Epochs"),
        hidden=mo.ui.slider(4, 64, value=16, step=4, label="Hidden size"),
        optimizer=mo.ui.dropdown(
            options={"SGD": "sgd", "SGD+Momentum": "sgd_mom", "Adam": "adam"},
            value="Adam",
            label="Optimizer",
        ),
        weight_decay=mo.ui.slider(0.0, 0.1, value=0.0, step=0.005,
                                      label="Weight decay (L2)"),
        dropout=mo.ui.slider(0.0, 0.8, value=0.0, step=0.1, label="Dropout"),
    ).form(submit_button_label="🚀 Train")
    train_form




    return (train_form,)


@app.cell
def _(mo, nn, np, plt, torch, train_form):
    mo.stop(train_form.value is None,
            mo.md("*Adjust hyperparameters above and click **Train** to start.*"))

    _cfg = train_form.value
    _lr = _cfg["lr"]
    _epochs = int(_cfg["epochs"])
    _hidden = int(_cfg["hidden"])
    _opt_name = _cfg["optimizer"]
    _wd = _cfg["weight_decay"]
    _drop = _cfg["dropout"]

    torch.manual_seed(0)
    np.random.seed(0)

    _n = 150
    _X = np.vstack([
        np.random.randn(_n, 2) * 0.5 + [2, 2],
        np.random.randn(_n, 2) * 0.5 + [-2, 2],
        np.random.randn(_n, 2) * 0.5 + [0, -2],
    ]).astype(np.float32)
    _Y = np.array([0] * _n + [1] * _n + [2] * _n, dtype=np.int64)

    _idx = np.random.permutation(len(_Y))
    _X, _Y = _X[_idx], _Y[_idx]

    _X_t = torch.from_numpy(_X)
    _Y_t = torch.from_numpy(_Y)

    _model = nn.Sequential(
        nn.Linear(2, _hidden),
        nn.ReLU(),
        nn.Dropout(_drop),
        nn.Linear(_hidden, _hidden),
        nn.ReLU(),
        nn.Dropout(_drop),
        nn.Linear(_hidden, 3),
    )
    _loss_fn = nn.CrossEntropyLoss()

    if _opt_name == "sgd":
        _opt = torch.optim.SGD(_model.parameters(), lr=_lr, weight_decay=_wd)
    elif _opt_name == "sgd_mom":
        _opt = torch.optim.SGD(_model.parameters(), lr=_lr, momentum=0.9,
                                weight_decay=_wd)
    else:
        _opt = torch.optim.Adam(_model.parameters(), lr=_lr, weight_decay=_wd)

    _losses = []
    _accs = []
    _model.train()
    for _ep in range(_epochs):
        _scores = _model(_X_t)
        _loss = _loss_fn(_scores, _Y_t)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()

        with torch.no_grad():
            _pred = _scores.argmax(dim=1)
            _acc = (_pred == _Y_t).float().mean().item()
        _losses.append(_loss.item())
        _accs.append(_acc)

    _model.eval()
    with torch.no_grad():
        _xx = np.linspace(-5, 5, 200)
        _yy = np.linspace(-5, 5, 200)
        _XX, _YY = np.meshgrid(_xx, _yy)
        _grid = torch.from_numpy(
            np.column_stack([_XX.ravel(), _YY.ravel()]).astype(np.float32)
        )
        _grid_pred = _model(_grid).argmax(dim=1).numpy().reshape(_XX.shape)

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))

    _cmap = plt.cm.RdYlGn
    _axes[0].contourf(_XX, _YY, _grid_pred, levels=[-0.5, 0.5, 1.5, 2.5],
                       colors=["#ffcccc", "#ccffcc", "#ccccff"], alpha=0.5)
    _scatter_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for _c in range(3):
        _mask = _Y == _c
        _axes[0].scatter(_X[_mask, 0], _X[_mask, 1], c=_scatter_colors[_c],
                          s=10, label=f"Class {_c}")
    _axes[0].set_title(f"Decision Boundaries (acc={_accs[-1]:.1%})")
    _axes[0].legend(fontsize=7)
    _axes[0].set_xlim(-5, 5)
    _axes[0].set_ylim(-5, 5)

    _axes[1].plot(_losses, color="#e67e22", linewidth=1.5)
    _axes[1].set_xlabel("Epoch")
    _axes[1].set_ylabel("CE Loss")
    _axes[1].set_title("Training Loss")
    if min(_losses) > 0:
        _axes[1].set_yscale("log")

    _axes[2].plot(_accs, color="#2ecc71", linewidth=1.5)
    _axes[2].set_xlabel("Epoch")
    _axes[2].set_ylabel("Accuracy")
    _axes[2].set_title("Training Accuracy")
    _axes[2].set_ylim(0, 1.05)

    plt.suptitle(
        f"Optimizer: {_opt_name} | LR: {_lr} | Hidden: {_hidden} | "
        f"WD: {_wd} | Dropout: {_drop}",
        fontsize=10,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Experiments to try:**
    1. Compare SGD vs Adam — which converges faster?
    2. Increase weight decay to 0.05 — does accuracy decrease (underfitting)?
    3. Set dropout to 0.5 — does training accuracy decrease? (It should — dropout
       adds noise during training. Test accuracy on unseen data would likely improve.)
    4. Reduce hidden size to 4 — can the network still learn 3 clusters?
    """)
    return


if __name__ == "__main__":
    app.run()
