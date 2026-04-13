# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib",
#     "numpy",
#     "torch",
# ]
# ///
"""
IME 775: Convolutions in Neural Networks — Interactive PyTorch Notebook
========================================================================
An interactive marimo notebook covering 1D, 2D, and 3D convolutions,
transposed convolution, and pooling with PyTorch.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 10 — Convolutions in Neural Networks
Topics: 1D/2D/3D Conv, Smoothing, Edge Detection, Transpose Conv, Pooling
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
    # IME 775: Convolutions in Neural Networks — Chapter 10 PyTorch Lab

    ## Learning Objectives

    1. Understand 1D convolution as a sliding weighted sum and implement it in PyTorch
    2. Apply 2D convolution for image smoothing and edge detection
    3. Implement 3D convolution for video motion detection
    4. Explore transposed convolution for upsampling
    5. Compare max pooling and average pooling

    ---

    | Section | Topic | Key Concept |
    |---------|-------|-------------|
    | 1 | 1D Convolution | Smoothing & Edge Detection |
    | 2 | Output Size Formula | $o = \lfloor(n+2p-k)/s\rfloor + 1$ |
    | 3 | 2D Convolution | Image Smoothing & Edge Detection |
    | 4 | 3D Convolution | Video Motion Detection |
    | 5 | Transposed Convolution | Upsampling |
    | 6 | Pooling | Max & Average Pooling |
    | 7 | CNN Building Block | Conv → ReLU → Pool |
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    return F, nn, np, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. One-Dimensional Convolution

    A 1D convolution slides a kernel over an input array, computing a weighted sum
    at each position. The kernel weights determine what local pattern is extracted.
    """)
    return


@app.cell
def _(mo):
    kernel_type_1d = mo.ui.dropdown(
        options={"Smoothing [1/3, 1/3, 1/3]": "smooth", "Edge Detection [0.5, -0.5]": "edge"},
        value="smooth",
        label="1D Kernel Type",
    )
    stride_1d = mo.ui.slider(1, 3, value=1, label="Stride")
    mo.hstack([kernel_type_1d, stride_1d], gap=1)
    return kernel_type_1d, stride_1d


@app.cell
def _(kernel_type_1d, nn, np, plt, stride_1d, torch):
    if kernel_type_1d.value == "smooth":
        _x = torch.tensor([14.0, -1.0, 4.0, 11.0, 21.0, 25.0, 30.0])
        _w = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        _title = "1D Smoothing Convolution (kernel=[1/3, 1/3, 1/3])"
    else:
        _x = torch.tensor(
            [10.0, 10.0, 10.0, 10.0, 51.0, 51.0, 51.0, 51.0, 49.0, 9.0, 9.0]
        )
        _w = torch.tensor([0.5, -0.5])
        _title = "1D Edge Detection (kernel=[0.5, -0.5])"

    _k = len(_w)
    _s = stride_1d.value
    _x4 = _x.unsqueeze(0).unsqueeze(0)
    _w4 = _w.unsqueeze(0).unsqueeze(0)

    _conv = nn.Conv1d(1, 1, kernel_size=_k, stride=_s, bias=False)
    _conv.weight = nn.Parameter(_w4, requires_grad=False)
    with torch.no_grad():
        _y = _conv(_x4).squeeze()

    _fig, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.plot(np.arange(len(_x)), _x.numpy(), "o-", label="Input", color="#3b82f6")
    _out_idx = np.arange(len(_y)) * _s + (_k - 1) / 2
    _ax.plot(_out_idx, _y.numpy(), "s--", label="Output", color="#f97316")
    _ax.set_title(_title, fontsize=13)
    _ax.set_xlabel("Index")
    _ax.set_ylabel("Value")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Output Size Formula

    $$o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$
    """)
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(5, 64, value=32, label="Input size n")
    k_slider = mo.ui.slider(1, 11, value=5, step=2, label="Kernel size k")
    s_slider = mo.ui.slider(1, 4, value=1, label="Stride s")
    p_slider = mo.ui.slider(0, 5, value=0, label="Padding p")
    mo.hstack([n_slider, k_slider, s_slider, p_slider], gap=1)
    return k_slider, n_slider, p_slider, s_slider


@app.cell
def _(k_slider, mo, n_slider, p_slider, s_slider):
    _n = n_slider.value
    _k = k_slider.value
    _s = s_slider.value
    _p = p_slider.value
    _o = (_n + 2 * _p - _k) // _s + 1
    mo.md(
        f"**Output size:** $o = \\lfloor({_n} + 2 \\times {_p} - {_k}) / {_s}\\rfloor + 1 = {_o}$"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Two-Dimensional Convolution

    2D convolution slides a 2D kernel (tile) over a 2D input (wall), essential for
    image processing. Different kernels extract different features.
    """)
    return


@app.cell
def _(mo):
    kernel_type_2d = mo.ui.dropdown(
        options={
            "Smoothing 3×3": "smooth",
            "Vertical Edge 2×2": "vedge",
            "Horizontal Edge 2×2": "hedge",
        },
        value="smooth",
        label="2D Kernel Type",
    )
    kernel_type_2d
    return (kernel_type_2d,)


@app.cell
def _(kernel_type_2d, nn, np, plt, torch):
    _input_smooth = torch.tensor(
        [
            [0.0, 6.0, 12.0, 18.0, 23.0],
            [12.0, 19.0, 25.0, 31.0, 37.0],
            [26.0, 31.0, 38.0, 43.0, 49.0],
            [39.0, 44.0, 50.0, 57.0, 63.0],
            [51.0, 57.0, 63.0, 70.0, 75.0],
        ]
    )
    _input_edge = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0],
            [10.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 100.0, 100.0],
        ]
    )

    if kernel_type_2d.value == "smooth":
        _x = _input_smooth
        _w = torch.full((3, 3), 1 / 9)
        _label = "Smoothing (3×3 uniform)"
    elif kernel_type_2d.value == "vedge":
        _x = _input_edge
        _w = torch.tensor([[-0.25, 0.25], [-0.25, 0.25]])
        _label = "Vertical Edge Detection"
    else:
        _x = _input_edge
        _w = torch.tensor([[-0.25, -0.25], [0.25, 0.25]])
        _label = "Horizontal Edge Detection"

    _k = _w.shape[0]
    _x4 = _x.unsqueeze(0).unsqueeze(0)
    _w4 = _w.unsqueeze(0).unsqueeze(0)
    _conv = nn.Conv2d(1, 1, kernel_size=_k, stride=1, bias=False)
    _conv.weight = nn.Parameter(_w4, requires_grad=False)
    with torch.no_grad():
        _y = _conv(_x4).squeeze()

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4))
    _ax1.imshow(_x.numpy(), cmap="gray")
    _ax1.set_title("Input")
    _ax2.imshow(_w.numpy(), cmap="RdBu_r")
    _ax2.set_title(f"Kernel: {_label}")
    for _i in range(_w.shape[0]):
        for _j in range(_w.shape[1]):
            _ax2.text(_j, _i, f"{_w[_i,_j]:.2f}", ha="center", va="center", fontsize=9)
    _ax3.imshow(_y.numpy(), cmap="gray")
    _ax3.set_title("Output")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Three-Dimensional Convolution (Motion Detection)

    3D convolution slides a brick-shaped kernel through a spatio-temporal volume.
    Below, we create a synthetic video of a moving block and detect its motion.
    """)
    return


@app.cell
def _(nn, np, plt, torch):
    _frames = torch.zeros(5, 32, 32)
    for _t in range(5):
        _r = 4 + _t * 3
        _c = 4 + _t * 3
        _frames[_t, _r : _r + 8, _c : _c + 8] = 200.0

    _w2d = torch.ones(1, 3, 3)
    _w3d = torch.cat([-_w2d, _w2d], dim=0).unsqueeze(0).unsqueeze(0)

    _x5d = _frames.unsqueeze(0).unsqueeze(0)
    _conv3d = nn.Conv3d(1, 1, kernel_size=(2, 3, 3), stride=1, padding=0, bias=False)
    _conv3d.weight = nn.Parameter(_w3d, requires_grad=False)
    with torch.no_grad():
        _y = _conv3d(_x5d).squeeze()

    _fig, _axes = plt.subplots(2, 4, figsize=(14, 7))
    for _i in range(4):
        _axes[0, _i].imshow(_frames[_i].numpy(), cmap="gray", vmin=0, vmax=255)
        _axes[0, _i].set_title(f"Input frame {_i}")
        _axes[0, _i].axis("off")
        _axes[1, _i].imshow(
            _y[_i].numpy(), cmap="RdBu_r", vmin=-_y.abs().max(), vmax=_y.abs().max()
        )
        _axes[1, _i].set_title(f"Motion frame {_i}")
        _axes[1, _i].axis("off")
    plt.suptitle("3D Convolution: Motion Detection", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Transposed Convolution (Upsampling)

    Transposed convolution reverses the spatial effect of convolution —
    it maps a smaller input to a larger output. This is used in decoder
    networks and autoencoders.
    """)
    return


@app.cell
def _(nn, np, plt, torch):
    _x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _w = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    _x4 = _x.unsqueeze(0).unsqueeze(0)
    _w4 = _w.unsqueeze(0).unsqueeze(0)

    _tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    _tconv.weight = nn.Parameter(_w4, requires_grad=False)
    with torch.no_grad():
        _y = _tconv(_x4).squeeze()

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(12, 4))
    _ax1.imshow(_x.numpy(), cmap="viridis")
    _ax1.set_title(f"Input (2×2)")
    for _i in range(2):
        for _j in range(2):
            _ax1.text(_j, _i, f"{_x[_i,_j]:.0f}", ha="center", va="center",
                      color="white", fontsize=14, fontweight="bold")

    _ax2.imshow(_w.numpy(), cmap="viridis")
    _ax2.set_title("Kernel (2×2)")
    for _i in range(2):
        for _j in range(2):
            _ax2.text(_j, _i, f"{_w[_i,_j]:.0f}", ha="center", va="center",
                      color="white", fontsize=14, fontweight="bold")

    _ax3.imshow(_y.numpy(), cmap="viridis")
    _ax3.set_title(f"Output (4×4)")
    for _i in range(4):
        for _j in range(4):
            _ax3.text(_j, _i, f"{_y[_i,_j]:.0f}", ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
    plt.suptitle("Transposed Convolution: stride=2, kernel=2×2", fontsize=13)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Pooling

    Pooling downsamples feature maps for local translation invariance.
    """)
    return


@app.cell
def _(mo):
    pool_type = mo.ui.dropdown(
        options={"Max Pooling": "max", "Average Pooling": "avg"},
        value="max",
        label="Pooling Type",
    )
    pool_type
    return (pool_type,)


@app.cell
def _(nn, np, plt, pool_type, torch):
    _x = torch.tensor(
        [
            [31.0, 43.0, 57.0, 70.0],
            [25.0, 38.0, 50.0, 63.0],
            [19.0, 31.0, 44.0, 57.0],
            [12.0, 26.0, 39.0, 51.0],
        ]
    ).unsqueeze(0).unsqueeze(0)

    if pool_type.value == "max":
        _pool = nn.MaxPool2d(kernel_size=2, stride=2)
        _label = "Max Pooling"
    else:
        _pool = nn.AvgPool2d(kernel_size=2, stride=2)
        _label = "Average Pooling"

    _y = _pool(_x).squeeze()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.imshow(_x.squeeze().numpy(), cmap="viridis")
    _ax1.set_title("Input (4×4)")
    for _i in range(4):
        for _j in range(4):
            _ax1.text(_j, _i, f"{_x.squeeze()[_i,_j]:.0f}", ha="center", va="center",
                      color="white", fontsize=10, fontweight="bold")
    _ax2.imshow(_y.numpy(), cmap="viridis")
    _ax2.set_title(f"{_label} Output (2×2)")
    for _i in range(2):
        for _j in range(2):
            _ax2.text(_j, _i, f"{_y[_i,_j]:.1f}", ha="center", va="center",
                      color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7. Full CNN Building Block: Conv → ReLU → Pool

    A typical CNN chains convolution, activation, and pooling layers.
    Below we construct a small CNN and visualize the intermediate feature maps.
    """)
    return


@app.cell
def _(nn, np, plt, torch):
    _model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )

    _x = torch.randn(1, 1, 28, 28)
    _activations = []
    _labels = []
    _out = _x
    for _name, _layer in _model.named_children():
        _out = _layer(_out)
        _activations.append(_out.detach().squeeze())
        _labels.append(f"{_layer.__class__.__name__} → {list(_out.shape[1:])}")

    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8))
    _axes[0, 0].imshow(_x.squeeze().numpy(), cmap="gray")
    _axes[0, 0].set_title(f"Input [1, 28, 28]")
    _axes[0, 0].axis("off")

    _show_indices = [0, 1, 2, 3, 4]
    for _idx, _si in enumerate(_show_indices):
        _r, _c = (_idx + 1) // 3, (_idx + 1) % 3
        _act = _activations[_si]
        if _act.dim() == 3:
            _act = _act[0]
        _axes[_r, _c].imshow(_act.numpy(), cmap="viridis")
        _axes[_r, _c].set_title(_labels[_si], fontsize=10)
        _axes[_r, _c].axis("off")
    plt.suptitle("Feature Maps Through a CNN Pipeline", fontsize=14)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Concept | Key Equation / Idea |
    |---------|---------------------|
    | 1D Conv | $Y_x = \sum_j X_{x+j} W_j$ |
    | Output size | $o = \lfloor(n + 2p - k)/s\rfloor + 1$ |
    | 2D Conv | $Y_{y,x} = \sum_i \sum_j X_{y+i,x+j} W_{i,j}$ |
    | 3D Conv | Adds temporal dimension for video analysis |
    | Transposed Conv | $\tilde{x} = W^T \vec{y}$ — upsampling |
    | Pooling | Max or Average over local patches — translation invariance |
    | CNN pipeline | Conv → ReLU → Pool (repeat) → FC → Softmax |
    """)
    return


if __name__ == "__main__":
    app.run()
