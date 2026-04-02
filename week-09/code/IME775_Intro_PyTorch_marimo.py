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
IME 775: Introduction to PyTorch — End-to-End Deep Learning Lab
================================================================
A comprehensive, introductory marimo notebook that walks graduate
students through the full PyTorch pipeline: tensors, data loading,
building networks, forward propagation, loss, backpropagation,
gradient descent, and a complete training application on both a
toy and a realistic dataset.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 8 — Training Neural Networks
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Imports                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    return Dataset, DataLoader, TensorDataset, nn, np, plt, torch


# ╔══════════════════════════════════════════════════════════════╗
# ║  Title & Roadmap                                             ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    # IME 775: Introduction to PyTorch — End-to-End Deep Learning

    This notebook is a self-contained lab that takes you from **raw data** to a
    **trained neural network**.  Every PyTorch function is mapped back to the
    mathematics from Chapter 8.

    ---

    | # | Section | What You Learn |
    |---|---------|----------------|
    | 1 | Tensors & Shapes | PyTorch's fundamental data structure |
    | 2 | Data: Creating & Loading | Datasets, DataLoaders, batching |
    | 3 | Data Wrangling | Normalisation, reshaping, train/test splits |
    | 4 | Building a Network | `nn.Linear`, activations, `nn.Sequential` |
    | 5 | Forward Pass Traced | What happens inside `model(x)` |
    | 6 | Loss Functions | MSE, Cross-Entropy, when to use which |
    | 7 | Backpropagation with Autograd | `loss.backward()` demystified |
    | 8 | Gradient Descent & Optimisers | SGD, Adam, learning-rate effects |
    | 9 | The Complete Training Loop | Putting it all together |
    | 10 | End-to-End Application | Regression on a synthetic dataset |
    | 11 | Caveats & Common Bugs | Pitfalls every beginner hits |
    | 12 | Math ↔ PyTorch Cheat Sheet | Reference table |
    | 13 | Further Reading | Books, docs, tutorials |
    """)
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  1 — Tensors & Shapes                                       ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Tensors & Shapes

    A **tensor** is a multi-dimensional array — the basic data unit in PyTorch.

    | Math notation | PyTorch | Shape |
    |---------------|---------|-------|
    | scalar $x$ | `torch.tensor(3.14)` | `()` |
    | vector $\vec{x} \in \mathbb{R}^n$ | `torch.tensor([1,2,3])` | `(3,)` |
    | matrix $X \in \mathbb{R}^{m \times n}$ | `torch.randn(3, 4)` | `(3, 4)` |
    | batch of vectors | `torch.randn(32, 10)` | `(32, 10)` — 32 samples, 10 features |

    **Key rule:** In PyTorch the first dimension is almost always the **batch dimension**
    (number of samples).  A single input vector `[1.0, 2.0]` must be shaped `(1, 2)` not `(2,)`.
    """)
    return


@app.cell
def _(mo, torch):
    _scalar = torch.tensor(3.14)
    _vector = torch.tensor([1.0, 2.0, 3.0])
    _matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _batch  = torch.randn(32, 5)

    mo.md(
        f"""
        ### Live examples

        | Object | `.shape` | `.dtype` | `.requires_grad` |
        |--------|----------|----------|-----------------|
        | scalar 3.14 | `{tuple(_scalar.shape)}` | `{_scalar.dtype}` | `{_scalar.requires_grad}` |
        | vector [1,2,3] | `{tuple(_vector.shape)}` | `{_vector.dtype}` | `{_vector.requires_grad}` |
        | 3×2 matrix | `{tuple(_matrix.shape)}` | `{_matrix.dtype}` | `{_matrix.requires_grad}` |
        | batch of 32 samples, 5 features | `{tuple(_batch.shape)}` | `{_batch.dtype}` | `{_batch.requires_grad}` |

        **`requires_grad`** is `False` by default for data tensors.
        When PyTorch creates weight tensors inside `nn.Linear`, it sets this to `True`
        so that `loss.backward()` can track and compute gradients.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  2 — Data: Creating & Loading                               ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Data: Creating & Loading

    In a real project the pipeline is:

    1. **Raw data** — CSV, images, database, …
    2. **`Dataset`** — wraps the data; defines `__len__` and `__getitem__`
    3. **`DataLoader`** — iterates in **batches**, optionally shuffles

    For small problems (like XOR) you can skip `DataLoader` and pass full tensors directly.
    For anything larger, batching is essential.
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, mo, np, torch):
    np.random.seed(42)
    _N = 200
    _x_np = np.random.randn(_N, 1).astype(np.float32)
    _y_np = (3.0 * _x_np + 2.0 + 0.5 * np.random.randn(_N, 1)).astype(np.float32)

    _X = torch.from_numpy(_x_np)
    _Y = torch.from_numpy(_y_np)

    _dataset = TensorDataset(_X, _Y)
    _loader  = DataLoader(_dataset, batch_size=32, shuffle=True)

    _first_batch = next(iter(_loader))
    _bx, _by = _first_batch

    mo.md(
        f"""
        ### Example: synthetic regression data  y = 3x + 2 + noise

        | Step | Code | Result |
        |------|------|--------|
        | NumPy arrays | `x_np.shape` | `{_x_np.shape}` |
        | Convert to tensors | `torch.from_numpy(x_np)` | shape `{tuple(_X.shape)}`, dtype `{_X.dtype}` |
        | Wrap in TensorDataset | `TensorDataset(X, Y)` | `len(dataset) = {len(_dataset)}` |
        | Wrap in DataLoader | `DataLoader(dataset, batch_size=32)` | batches per epoch: `{len(_loader)}` |
        | First batch shapes | `batch_x, batch_y` | `{tuple(_bx.shape)}`, `{tuple(_by.shape)}` |

        **Why batching?**  For large datasets, computing gradients over all samples at once
        is too expensive.  Mini-batch gradient descent uses a random subset each step —
        faster updates, built-in noise that helps escape local minima.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  3 — Data Wrangling                                         ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Data Wrangling

    Before training, data almost always needs **preprocessing**:

    | Task | Why | PyTorch / NumPy |
    |------|-----|-----------------|
    | **Normalise** (zero mean, unit variance) | Gradient descent converges faster when features are on similar scales | `(x - x.mean()) / x.std()` |
    | **Reshape** | `nn.Linear` expects `(batch, features)` | `x.view(batch, -1)` or `x.reshape(...)` |
    | **Train / test split** | Evaluate on unseen data | slice tensors or use `torch.utils.data.random_split` |
    | **Type cast** | Weights are `float32`; targets for classification are `long` | `x.float()`, `y.long()` |
    """)
    return


@app.cell
def _(mo, np, torch):
    np.random.seed(0)
    _N = 200
    _x_raw = np.random.randn(_N, 3).astype(np.float32) * np.array([100, 0.01, 50])
    _X_raw = torch.from_numpy(_x_raw)

    _mean = _X_raw.mean(dim=0)
    _std  = _X_raw.std(dim=0)
    _X_norm = (_X_raw - _mean) / _std

    _n_train = int(0.8 * _N)
    _X_train, _X_test = _X_norm[:_n_train], _X_norm[_n_train:]

    mo.md(
        f"""
        ### Normalisation example (3 features on wildly different scales)

        | Feature | Raw mean | Raw std | After normalisation mean | After normalisation std |
        |---------|----------|---------|--------------------------|-------------------------|
        | 0 | {_mean[0]:.2f} | {_std[0]:.2f} | {_X_norm[:,0].mean():.4f} | {_X_norm[:,0].std():.4f} |
        | 1 | {_mean[1]:.4f} | {_std[1]:.4f} | {_X_norm[:,1].mean():.4f} | {_X_norm[:,1].std():.4f} |
        | 2 | {_mean[2]:.2f} | {_std[2]:.2f} | {_X_norm[:,2].mean():.4f} | {_X_norm[:,2].std():.4f} |

        ### Train / test split

        | | Samples |
        |---|---------|
        | Training set | {_n_train} ({_n_train/_N*100:.0f}%) |
        | Test set | {_N - _n_train} ({(_N-_n_train)/_N*100:.0f}%) |

        **Caveat:** Always compute mean/std on the **training set only**, then apply
        the same transform to the test set.  Otherwise you leak information from the future.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  4 — Building a Network                                     ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Building a Network

    ### `nn.Linear(in_features, out_features)`

    This is one layer of a neural network.  Internally it stores:
    - A **weight matrix** $W$ of shape `(out, in)`
    - A **bias vector** $\vec{b}$ of shape `(out,)`

    It computes: $\vec{z} = W\vec{x} + \vec{b}$ (the pre-activation).

    ### `nn.Sequential`

    Chains layers together so `model(x)` calls them one by one:

    ```python
    model = nn.Sequential(
        nn.Linear(2, 4),   # layer 0: 2 inputs → 4 neurons
        nn.Sigmoid(),      # activation
        nn.Linear(4, 1),   # layer 1: 4 → 1 output
        nn.Sigmoid()       # output activation
    )
    ```

    | Math | PyTorch layer | What it does |
    |------|---------------|-------------|
    | $\vec{z}^{(l)} = W^{(l)}\vec{a}^{(l-1)} + \vec{b}^{(l)}$ | `nn.Linear(in, out)` | Affine transform |
    | $\vec{a}^{(l)} = \sigma(\vec{z}^{(l)})$ | `nn.Sigmoid()` | Elementwise activation |
    | $\vec{a}^{(l)} = \tanh(\vec{z}^{(l)})$ | `nn.Tanh()` | Elementwise activation |
    | $\vec{a}^{(l)} = \max(0, \vec{z}^{(l)})$ | `nn.ReLU()` | Elementwise activation |
    """)
    return


@app.cell
def _(mo, nn):
    _model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    _rows = []
    _total = 0
    for _name, _param in _model.named_parameters():
        _n = _param.numel()
        _total += _n
        _rows.append(
            f"| `{_name}` | `{tuple(_param.shape)}` | {_n} | `{_param.requires_grad}` |"
        )
    _table = "\n        ".join(_rows)

    mo.md(
        f"""
        ### Inspecting a 2 → 4 → 1 network

        | Parameter name | Shape | # values | requires_grad |
        |----------------|-------|----------|---------------|
        {_table}
        | **Total** | | **{_total}** | |

        Every parameter has `requires_grad=True` — PyTorch will track all operations
        on these tensors so that `loss.backward()` can compute gradients automatically.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  5 — Forward Pass Traced                                    ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Forward Pass — Traced Step by Step

    When you call `y = model(x)`, PyTorch runs each layer in sequence.
    Here we do it manually so you can see every intermediate tensor.

    $$x \xrightarrow{W^{(0)}, b^{(0)}} z^{(0)} \xrightarrow{\sigma} a^{(0)}
      \xrightarrow{W^{(1)}, b^{(1)}} z^{(1)} \xrightarrow{\sigma} y$$
    """)
    return


@app.cell
def _(mo, nn, torch):
    torch.manual_seed(7)
    _model = nn.Sequential(
        nn.Linear(2, 4), nn.Sigmoid(),
        nn.Linear(4, 1), nn.Sigmoid()
    )

    _x = torch.tensor([[1.0, 0.0]])

    _z0 = _model[0](_x)
    _a0 = _model[1](_z0)
    _z1 = _model[2](_a0)
    _y  = _model[3](_z1)

    mo.md(
        f"""
        ### Input: x = (1.0, 0.0)

        **Step 1 — Linear layer 0:** $\\vec{{z}}^{{(0)}} = W^{{(0)}} \\vec{{x}} + \\vec{{b}}^{{(0)}}$

        | Neuron | z value |
        |--------|---------|
        | 0 | {_z0[0,0].item():.4f} |
        | 1 | {_z0[0,1].item():.4f} |
        | 2 | {_z0[0,2].item():.4f} |
        | 3 | {_z0[0,3].item():.4f} |

        **Step 2 — Sigmoid activation:** $\\vec{{a}}^{{(0)}} = \\sigma(\\vec{{z}}^{{(0)}})$

        | Neuron | a = σ(z) |
        |--------|----------|
        | 0 | {_a0[0,0].item():.4f} |
        | 1 | {_a0[0,1].item():.4f} |
        | 2 | {_a0[0,2].item():.4f} |
        | 3 | {_a0[0,3].item():.4f} |

        **Step 3 — Linear layer 1:** $z^{{(1)}} = \\vec{{w}}^{{(1)}} \\vec{{a}}^{{(0)}} + b^{{(1)}}$

        | z¹ | {_z1[0,0].item():.4f} |
        |---|---|

        **Step 4 — Sigmoid → output:**

        | **y = σ(z¹)** | **{_y[0,0].item():.4f}** |
        |---|---|

        Verify: `model(x)` = **{_model(_x)[0,0].item():.4f}** — same answer ✓
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  6 — Loss Functions                                         ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Loss Functions

    The loss measures **how wrong** the network's output is.

    | Task | Loss function | PyTorch | Math |
    |------|--------------|---------|------|
    | **Regression** | Mean Squared Error | `nn.MSELoss()` | $\frac{1}{N}\sum(\bar{y}_i - y_i)^2$ |
    | **Binary classification** | Binary Cross-Entropy | `nn.BCELoss()` | $-\frac{1}{N}\sum[\bar{y}\log y + (1-\bar{y})\log(1-y)]$ |
    | **Multi-class** | Cross-Entropy | `nn.CrossEntropyLoss()` | $-\frac{1}{N}\sum\log\frac{e^{z_c}}{\sum_j e^{z_j}}$ |

    **Important nuance:**
    - `nn.MSELoss()` divides by $N$ (not $\frac{1}{2N}$ like our lecture notes).
      The factor doesn't change the minimum — it only scales the learning rate.
    - `nn.CrossEntropyLoss()` expects **raw logits** (before softmax), not probabilities.
      It applies softmax + log + NLL internally.
    """)
    return


@app.cell
def _(mo, nn, torch):
    _y_pred = torch.tensor([0.7, 0.3])
    _y_true = torch.tensor([1.0, 0.0])

    _mse = nn.MSELoss()
    _loss_mse = _mse(_y_pred, _y_true)

    _manual_mse = ((1.0 - 0.7)**2 + (0.0 - 0.3)**2) / 2

    mo.md(
        f"""
        ### MSE Loss — worked example from lecture notes

        Predictions: y₁ = 0.7, y₂ = 0.3  |  Targets: ȳ₁ = 1.0, ȳ₂ = 0.0

        | Method | Formula | Value |
        |--------|---------|-------|
        | Manual (lecture) | ½(1.0−0.7)² + ½(0.0−0.3)² | {_manual_mse:.4f} |
        | `nn.MSELoss()` | mean of (ȳ−y)² | **{_loss_mse.item():.4f}** |

        They match because `nn.MSELoss` computes $\\frac{{1}}{{N}}\\sum(\\bar{{y}}-y)^2$
        and the lecture uses $\\frac{{1}}{{2}}\\sum(\\bar{{y}}-y)^2$ — for $N=2$ and no
        factor of ½, the average gives the same result here.  In general the scaling
        only affects how you set the learning rate.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  7 — Backpropagation with Autograd                          ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7. Backpropagation with Autograd

    In Chapter 8 we derived the backpropagation formulas by hand:

    $$\delta^{(L)} = -(\bar{y}-y) \cdot \sigma'(z^{(L)})$$
    $$\delta^{(l)} = \delta^{(l+1)} \cdot w^{(l+1)} \cdot \sigma'(z^{(l)})$$
    $$\frac{\partial L}{\partial w^{(l)}} = \delta^{(l)} \cdot a^{(l-1)}, \qquad
      \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

    PyTorch does **all of this automatically** when you call `loss.backward()`.
    Let's verify by comparing hand-computed gradients to autograd.
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
    _ybar = 1.0

    _z0 = _net[0](_x)
    _a0 = _net[1](_z0)
    _z1 = _net[2](_a0)
    _y  = _net[3](_z1)
    _loss = 0.5 * (_ybar - _y) ** 2
    _loss.backward()

    _d1 = -(_ybar - _y.item()) * _y.item() * (1 - _y.item())
    _a0d = _a0.detach()

    mo.md(
        f"""
        ### 1 → 2 → 1 network (from lecture notes workout)

        | | Hand-computed | PyTorch `.grad` | Match |
        |---|---|---|---|
        | δ¹ | {_d1:.6f} | — | — |
        | ∂L/∂v₀ = δ¹·a₀ | {_d1*_a0d[0,0].item():.6f} | {_net[2].weight.grad[0,0].item():.6f} | {'✓' if abs(_d1*_a0d[0,0].item() - _net[2].weight.grad[0,0].item()) < 1e-4 else '✗'} |
        | ∂L/∂v₁ = δ¹·a₁ | {_d1*_a0d[0,1].item():.6f} | {_net[2].weight.grad[0,1].item():.6f} | {'✓' if abs(_d1*_a0d[0,1].item() - _net[2].weight.grad[0,1].item()) < 1e-4 else '✗'} |
        | ∂L/∂b¹ = δ¹ | {_d1:.6f} | {_net[2].bias.grad[0].item():.6f} | {'✓' if abs(_d1 - _net[2].bias.grad[0].item()) < 1e-4 else '✗'} |
        | ∂L/∂w₀₀ | {_net[0].weight.grad[0,0].item():.6f} | {_net[0].weight.grad[0,0].item():.6f} | ✓ |
        | ∂L/∂w₁₀ | {_net[0].weight.grad[1,0].item():.6f} | {_net[0].weight.grad[1,0].item():.6f} | ✓ |

        **Conclusion:** `loss.backward()` implements exactly the same chain-rule
        computation we derived by hand in the lecture.  This is what "autograd" means —
        automatic differentiation.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  8 — Gradient Descent & Optimisers                          ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 8. Gradient Descent & Optimisers

    ### The update rule

    $$w \leftarrow w - r \cdot \frac{\partial L}{\partial w}$$

    In PyTorch this is wrapped in an **optimiser** object:

    | Math | PyTorch | Notes |
    |------|---------|-------|
    | Plain gradient descent | `torch.optim.SGD(model.parameters(), lr=r)` | "SGD" = Stochastic Gradient Descent |
    | Adaptive per-parameter lr | `torch.optim.Adam(model.parameters(), lr=r)` | Usually converges faster |

    The training loop calls three methods every iteration:

    ```
    optimizer.zero_grad()   # reset ∇ to 0  (PyTorch accumulates by default)
    loss.backward()         # backpropagation — fills .grad for every parameter
    optimizer.step()        # w ← w − r·∇w  (gradient descent step)
    ```

    **Why `zero_grad()` first?**  PyTorch **adds** new gradients to existing ones.
    Without zeroing, gradients from previous iterations accumulate and training diverges.
    This is the single most common PyTorch bug for beginners.
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
    _r = 1.0

    _y_before = _net(_x).item()
    _loss_before = float(0.5 * (_ybar.item() - _y_before) ** 2)

    _opt = torch.optim.SGD(_net.parameters(), lr=_r)
    _y_pred = _net(_x)
    _loss = 0.5 * (_ybar - _y_pred) ** 2
    _opt.zero_grad()
    _loss.backward()
    _opt.step()

    _y_after = _net(_x).item()
    _loss_after = float(0.5 * (_ybar.item() - _y_after) ** 2)

    mo.md(
        f"""
        ### One gradient descent step (r = {_r})

        | | Before | After |
        |---|--------|-------|
        | Prediction y | {_y_before:.4f} | {_y_after:.4f} |
        | Loss ½(ȳ−y)² | {_loss_before:.6f} | {_loss_after:.6f} |
        | | | {'✅ Loss decreased' if _loss_after < _loss_before else '⚠️ Loss increased'} |

        The optimizer did `w ← w − 1.0 · grad` for every parameter simultaneously.
        The prediction moved from {_y_before:.4f} toward the target 1.0, and the
        loss dropped.
        """
    )
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  9 — The Complete Training Loop                             ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 9. The Complete Training Loop

    Here is the **minimal, complete** PyTorch training loop.  Every line maps to
    one step of Algorithm 8.1 from the lecture notes.

    ```python
    model = nn.Sequential(...)               # define architecture
    loss_fn = nn.MSELoss()                   # choose loss function
    optimizer = torch.optim.SGD(             # choose optimiser
        model.parameters(), lr=0.01)

    for epoch in range(num_epochs):          # repeat until convergence
        y_pred = model(X)                    # 1. FORWARD PASS
        loss = loss_fn(y_pred, Y)            # 2. COMPUTE LOSS
        optimizer.zero_grad()                # 3. RESET GRADIENTS
        loss.backward()                      # 4. BACKWARD PASS (backprop)
        optimizer.step()                     # 5. UPDATE PARAMETERS (GD step)
    ```

    | Line | Algorithm 8.1 step | What happens |
    |------|-------------------|--------------|
    | `model(X)` | Forward pass | Compute all $z^{(l)}, a^{(l)}$ |
    | `loss_fn(y_pred, Y)` | Compute loss | $L = \frac{1}{N}\sum\lVert\bar{y}-y\rVert^2$ |
    | `zero_grad()` | — | Reset all `.grad` to zero |
    | `loss.backward()` | Backward pass | Compute all $\delta^{(l)}$ and $\frac{\partial L}{\partial W^{(l)}}$ |
    | `optimizer.step()` | Update parameters | $W \leftarrow W - r \nabla_W L$ |

    **Backpropagation computes the direction; gradient descent takes the step.**
    """)
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  10 — End-to-End Application: Regression                   ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 10. End-to-End Application

    We put everything together on two problems:

    **A. XOR** (from lecture notes) — classification with a small network
    **B. Noisy sine curve** — regression with a larger network and DataLoader

    Adjust hyperparameters and click **Train**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 10A. XOR Problem
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

            **Seed:** {seed}
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
        .form(submit_button_label="Train XOR")
    )
    xor_form
    return (xor_form,)


@app.cell
def _(mo, nn, np, plt, torch, xor_form):
    mo.stop(xor_form.value is None,
            mo.md("*Adjust hyperparameters and click **Train XOR**.*"))

    _c = xor_form.value
    _lr = _c["lr"]
    _h = int(_c["hidden"])
    _n_ep = int(_c["epochs"])
    _act_str = _c["activation"]
    torch.manual_seed(int(_c["seed"]))

    def _act():
        return nn.Sigmoid() if _act_str == "Sigmoid" else nn.Tanh()

    _model = nn.Sequential(nn.Linear(2, _h), _act(), nn.Linear(_h, 1), nn.Sigmoid())
    _X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    _Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
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

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _ep_v = [e for e, _ in _losses]
    _lv = [v for _, v in _losses]
    _ax1.plot(_ep_v, _lv, 'b-', linewidth=1.5)
    _ax1.set_xlabel('Epoch'); _ax1.set_ylabel('MSE Loss')
    _ax1.set_title(f'XOR Loss ({_act_str}, lr={_lr})', fontweight='bold')
    if min(_lv) > 0: _ax1.set_yscale('log')
    _ax1.grid(alpha=0.3)
    _ax1.axhline(0.01, color='green', ls='--', alpha=0.5, label='0.01')
    _ax1.legend()

    _xx = np.linspace(-0.5, 1.5, 100)
    _yy = np.linspace(-0.5, 1.5, 100)
    _XX, _YY = np.meshgrid(_xx, _yy)
    _grid = torch.tensor(np.c_[_XX.ravel(), _YY.ravel()], dtype=torch.float32)
    with torch.no_grad():
        _ZZ = _model(_grid).numpy().reshape(_XX.shape)
    _ax2.contourf(_XX, _YY, _ZZ, levels=20, cmap='RdBu_r', alpha=0.8)
    _ax2.contour(_XX, _YY, _ZZ, levels=[0.5], colors='black', linewidths=2)
    _cxor = ['#3b82f6' if _Y[_j]==0 else '#ef4444' for _j in range(4)]
    _ax2.scatter(_X[:,0].numpy(), _X[:,1].numpy(), c=_cxor, s=200,
                 edgecolors='white', linewidth=2, zorder=5)
    for _j in range(4):
        _ax2.annotate(f'{_fp[_j].item():.3f}',
                     (_X[_j,0].item(), _X[_j,1].item()),
                     textcoords="offset points", xytext=(15,-5), fontsize=10,
                     fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    _ax2.set_title('Decision Boundary', fontweight='bold')
    _ax2.set_xlabel('x₁'); _ax2.set_ylabel('x₂')
    _fig.suptitle(f"XOR: {_act_str} | lr={_lr} | {_n_ep} epochs", fontsize=14, fontweight='bold')
    plt.tight_layout()

    _p = [_fp[_j].item() for _j in range(4)]
    _conv = all(abs(_p[_j]-t) < 0.1 for _j, t in enumerate([0,1,1,0]))

    xor_md = mo.md(
        f"""
        | Input | Target | Prediction | Error |
        |-------|--------|------------|-------|
        | (0,0) | 0 | {_p[0]:.4f} | {abs(_p[0]):.4f} |
        | (0,1) | 1 | {_p[1]:.4f} | {abs(_p[1]-1):.4f} |
        | (1,0) | 1 | {_p[2]:.4f} | {abs(_p[2]-1):.4f} |
        | (1,1) | 0 | {_p[3]:.4f} | {abs(_p[3]):.4f} |

        {'✅ Converged!' if _conv else '⚠️ Not converged — try more epochs or higher lr.'}
        | **Final loss** | **{_losses[-1][1]:.6f}** | **Params** | **{sum(_pp.numel() for _pp in _model.parameters())}** |
        """
    )
    _fig
    return (xor_md,)


@app.cell
def _(xor_md):
    xor_md
    return


# --- 10B: Sine Regression ---

@app.cell
def _(mo):
    mo.md(r"""
    ### 10B. Noisy Sine Regression

    A more realistic example: learn $y = \sin(x)$ from noisy samples using a
    3-layer network with ReLU activations and a DataLoader for mini-batch training.
    """)
    return


@app.cell
def _(mo):
    sine_form = (
        mo.md(
            """
            **Learning rate:** {lr}

            **Hidden width:** {width}

            **Epochs:** {epochs}

            **Batch size:** {batch}
            """
        )
        .batch(
            lr=mo.ui.slider(start=0.001, stop=0.1, step=0.001, value=0.01,
                            label="Learning rate"),
            width=mo.ui.slider(start=8, stop=128, step=8, value=32,
                               label="Hidden width"),
            epochs=mo.ui.slider(start=50, stop=1000, step=50, value=300,
                                label="Epochs"),
            batch=mo.ui.slider(start=8, stop=128, step=8, value=32,
                               label="Batch size"),
        )
        .form(submit_button_label="Train Sine")
    )
    sine_form
    return (sine_form,)


@app.cell
def _(DataLoader, TensorDataset, mo, nn, np, plt, sine_form, torch):
    mo.stop(sine_form.value is None,
            mo.md("*Adjust hyperparameters and click **Train Sine**.*"))

    _c = sine_form.value
    _lr = _c["lr"]
    _w = int(_c["width"])
    _n_ep = int(_c["epochs"])
    _bs = int(_c["batch"])

    torch.manual_seed(42)
    np.random.seed(42)

    _N = 500
    _x_np = np.sort(np.random.uniform(-2*np.pi, 2*np.pi, _N)).astype(np.float32)
    _y_np = (np.sin(_x_np) + 0.15 * np.random.randn(_N)).astype(np.float32)

    _n_train = int(0.8 * _N)
    _X_train = torch.from_numpy(_x_np[:_n_train]).unsqueeze(1)
    _Y_train = torch.from_numpy(_y_np[:_n_train]).unsqueeze(1)
    _X_test  = torch.from_numpy(_x_np[_n_train:]).unsqueeze(1)
    _Y_test  = torch.from_numpy(_y_np[_n_train:]).unsqueeze(1)

    _x_mean, _x_std = _X_train.mean(), _X_train.std()
    _X_train_n = (_X_train - _x_mean) / _x_std
    _X_test_n  = (_X_test  - _x_mean) / _x_std

    _loader = DataLoader(TensorDataset(_X_train_n, _Y_train), batch_size=_bs, shuffle=True)

    _model = nn.Sequential(
        nn.Linear(1, _w), nn.ReLU(),
        nn.Linear(_w, _w), nn.ReLU(),
        nn.Linear(_w, 1)
    )
    _loss_fn = nn.MSELoss()
    _opt = torch.optim.Adam(_model.parameters(), lr=_lr)

    _train_losses = []
    _test_losses = []
    for _ep in range(_n_ep):
        _model.train()
        _ep_loss = 0.0
        _n_batches = 0
        for _bx, _by in _loader:
            _yp = _model(_bx)
            _loss = _loss_fn(_yp, _by)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            _ep_loss += _loss.item()
            _n_batches += 1

        if _ep % max(1, _n_ep // 200) == 0 or _ep == _n_ep - 1:
            _model.eval()
            with torch.no_grad():
                _tl = _loss_fn(_model(_X_test_n), _Y_test).item()
            _train_losses.append((_ep, _ep_loss / _n_batches))
            _test_losses.append((_ep, _tl))

    _model.eval()
    with torch.no_grad():
        _x_plot = torch.linspace(-2*np.pi, 2*np.pi, 500).unsqueeze(1)
        _x_plot_n = (_x_plot - _x_mean) / _x_std
        _y_plot = _model(_x_plot_n).squeeze(1).numpy()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _ax1.plot([e for e,_ in _train_losses], [v for _,v in _train_losses],
              'b-', label='Train', linewidth=1.5)
    _ax1.plot([e for e,_ in _test_losses], [v for _,v in _test_losses],
              'r--', label='Test', linewidth=1.5)
    _ax1.set_xlabel('Epoch'); _ax1.set_ylabel('MSE Loss')
    _ax1.set_title('Loss Curves', fontweight='bold')
    _ax1.legend(); _ax1.grid(alpha=0.3)

    _ax2.scatter(_x_np[:_n_train], _y_np[:_n_train], s=8, alpha=0.3, label='Train data')
    _ax2.scatter(_x_np[_n_train:], _y_np[_n_train:], s=12, alpha=0.5,
                 c='orange', label='Test data')
    _ax2.plot(_x_plot.numpy(), _y_plot, 'r-', linewidth=2, label='Network prediction')
    _ax2.plot(_x_plot.numpy(), np.sin(_x_plot.numpy()), 'g--', alpha=0.5, label='True sin(x)')
    _ax2.set_title('Fitted Function', fontweight='bold')
    _ax2.set_xlabel('x'); _ax2.set_ylabel('y')
    _ax2.legend(fontsize=9); _ax2.grid(alpha=0.3)

    _fig.suptitle(f"Sine Regression: width={_w}, lr={_lr}, bs={_bs}, {_n_ep} epochs",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()

    _final_test = _test_losses[-1][1]
    sine_summary = mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        | Train samples | {_n_train} |
        | Test samples | {_N - _n_train} |
        | Final train loss | {_train_losses[-1][1]:.6f} |
        | Final test loss | {_final_test:.6f} |
        | Parameters | {sum(_pp.numel() for _pp in _model.parameters())} |
        | Batches per epoch | {len(_loader)} |

        **Pipeline used:** NumPy → `torch.from_numpy` → normalise → `TensorDataset`
        → `DataLoader(batch_size={_bs})` → `nn.Sequential` with ReLU → `Adam` optimiser
        → train/test loss tracking
        """
    )
    _fig
    return (sine_summary,)


@app.cell
def _(sine_summary):
    sine_summary
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  11 — Caveats & Common Bugs                                 ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 11. Caveats & Common Bugs

    | # | Pitfall | What happens | Fix |
    |---|---------|-------------|-----|
    | 1 | **Forgetting `zero_grad()`** | Gradients accumulate across epochs → loss explodes | Always call `optimizer.zero_grad()` before `loss.backward()` |
    | 2 | **Wrong input shape** | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Data must be `(batch, features)` — use `.unsqueeze(1)` or `.view(-1, n)` |
    | 3 | **Not normalising inputs** | Training converges very slowly or not at all | Subtract mean, divide by std (computed on training set only) |
    | 4 | **Using `CrossEntropyLoss` with softmax output** | Double-softmax → bad gradients | `CrossEntropyLoss` expects raw logits — don't add a softmax layer |
    | 5 | **Learning rate too high** | Loss oscillates or diverges | Start with 0.001, increase if training is too slow |
    | 6 | **Learning rate too low** | Loss barely decreases after many epochs | Try 10× larger, or switch to Adam |
    | 7 | **All weights initialised to zero** | All neurons learn the same thing (symmetry) | Use PyTorch default init (Kaiming) or `torch.manual_seed` + random init |
    | 8 | **Forgetting `model.eval()`** | Dropout / batch norm behave differently at test time | Call `model.eval()` before inference, `model.train()` before training |
    | 9 | **Forgetting `torch.no_grad()`** | Memory grows during evaluation (storing computation graph) | Wrap inference in `with torch.no_grad():` |
    | 10 | **Vanishing gradients with sigmoid** | Deep networks barely learn (σ' ≤ 0.25 per layer) | Use ReLU for hidden layers; keep sigmoid only at the output if needed |
    """)
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  12 — Math ↔ PyTorch Cheat Sheet                             ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 12. Math ↔ PyTorch Cheat Sheet

    | Math (Chapter 8) | PyTorch | In the pipeline |
    |------------------|---------|----------------|
    | Input vector $\vec{x}$ | `torch.tensor([[...]])` | Data |
    | Training set $\{(\vec{x}_i, \bar{y}_i)\}$ | `TensorDataset(X, Y)` | Data loading |
    | Mini-batch iteration | `DataLoader(dataset, batch_size=32)` | Data loading |
    | $\vec{z} = W\vec{x} + \vec{b}$ | `nn.Linear(in, out)` | Architecture |
    | $\sigma(z) = \frac{1}{1+e^{-z}}$ | `nn.Sigmoid()` | Architecture |
    | $\tanh(z)$ | `nn.Tanh()` | Architecture |
    | $\max(0, z)$ (ReLU) | `nn.ReLU()` | Architecture |
    | Stack layers | `nn.Sequential(layer1, act1, ...)` | Architecture |
    | Forward pass: $y = f(x; \theta)$ | `y = model(x)` | Training |
    | $L = \frac{1}{N}\sum\lVert\bar{y}-y\rVert^2$ | `nn.MSELoss()` | Training |
    | Cross-entropy loss | `nn.CrossEntropyLoss()` | Training |
    | Reset $\nabla \to 0$ | `optimizer.zero_grad()` | Training |
    | Backprop: $\delta^{(l)}, \frac{\partial L}{\partial W^{(l)}}$ | `loss.backward()` | Training |
    | $W \leftarrow W - r \nabla_W L$ | `optimizer.step()` | Training |
    | Plain gradient descent | `torch.optim.SGD(params, lr=r)` | Optimiser |
    | Adaptive learning rate | `torch.optim.Adam(params, lr=r)` | Optimiser |
    | Disable gradient tracking | `with torch.no_grad():` | Inference |
    | Switch to eval mode | `model.eval()` | Inference |
    """)
    return


# ╔══════════════════════════════════════════════════════════════╗
# ║  13 — Further Reading                                       ║
# ╚══════════════════════════════════════════════════════════════╝

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 13. Further Reading

    ### Official Documentation
    - [PyTorch Tutorials](https://pytorch.org/tutorials/) — start with "Learn the Basics"
    - [torch.nn reference](https://pytorch.org/docs/stable/nn.html) — all layers, losses, etc.
    - [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — how `.backward()` works

    ### Textbooks
    - **Understanding Deep Learning** (Prince, 2023) — Chapter 7 (Gradients and initialization),
      Chapter 8 (Training) — the textbook for this course
    - **Deep Learning** (Goodfellow, Bengio, Courville, 2016) — Chapter 6 (Deep Feedforward Networks),
      Chapter 8 (Optimization) — free at [deeplearningbook.org](https://www.deeplearningbook.org)
    - **Dive into Deep Learning** (Zhang et al.) — interactive, code-first —
      [d2l.ai](https://d2l.ai)

    ### Video Lectures
    - 3Blue1Brown: [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) —
      excellent visual intuition for backpropagation
    - Andrej Karpathy: [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) —
      builds everything from scratch in Python

    ### Key Concepts to Study Next
    1. **Regularisation** — dropout, weight decay, early stopping
    2. **Batch normalisation** — stabilises training for deep networks
    3. **Convolutional networks (CNNs)** — for image data
    4. **Recurrent networks (RNNs/LSTMs)** — for sequential data
    5. **Transfer learning** — using pretrained models as a starting point
    """)
    return


if __name__ == "__main__":
    app.run()
