# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
#     "torch",
#     "torchvision",
# ]
# ///
"""
IME 775: Latent Spaces, Autoencoders, and Variational Autoencoders
==================================================================
An interactive marimo notebook that takes Chapter 14 of
"Math and Architectures of Deep Learning" from PCA to VAEs.

Sections
--------
1. PCA as a linear latent-space model (3D -> 2D)
2. Why PCA fails on curved manifolds (S-curve)
3. A convolutional autoencoder on MNIST
4. A variational autoencoder on MNIST
5. Latent-space comparison (AE vs VAE, nz=2)
6. Sampling from the VAE latent prior
7. KL-divergence explorer (closed-form Gaussian KL)
8. The reparameterization trick
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


# =============================================================================
# Cell 1: imports & title
# =============================================================================
@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # IME 775: Latent Spaces, Autoencoders, and VAEs — Chapter 14 Lab

    ## Learning Objectives

    1. See PCA as the **linear** case of latent-space modelling and implement it in PyTorch.
    2. Understand why PCA fails on curved manifolds by visualizing an S-curve.
    3. Train a convolutional **autoencoder** on MNIST and inspect its latent space.
    4. Train a **variational autoencoder (VAE)**, including the reparameterization trick and the KL term.
    5. Compare AE and VAE latent spaces and see why the VAE's regularization gives it **generative** power.

    | Section | Topic |
    |---|---|
    | 1 | PCA in PyTorch (3D → 2D) |
    | 2 | When PCA fails — the S-curve |
    | 3 | Convolutional autoencoder on MNIST |
    | 4 | VAE on MNIST (reparameterization + KL) |
    | 5 | AE vs VAE — 2D latent scatter |
    | 6 | Sampling from the VAE prior |
    | 7 | KL-divergence closed form |
    | 8 | Reparameterization-trick sanity check |

    > Training cells are configured to run quickly on CPU (few epochs, small batch).
    > Increase `EPOCHS` in the config cell for higher-quality results.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    np.random.seed(0)
    return F, nn, np, plt, torch


# =============================================================================
# Cell: global config
# =============================================================================
@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(1, 10, value=3, label="Training epochs (AE & VAE)")
    beta_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="VAE β (KL weight)")
    nz_slider = mo.ui.slider(2, 16, step=2, value=2, label="Latent dimension n_z")
    mo.md("### Global training configuration")
    mo.vstack([epochs_slider, beta_slider, nz_slider])
    return beta_slider, epochs_slider, nz_slider


# =============================================================================
# 1. PCA in PyTorch
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 1. PCA in PyTorch — linear latent space

    We generate 1,000 points clustered tightly around the plane $X_0 = X_2$ in $\mathbb{R}^3$ and run PCA via SVD.
    Dropping the smallest-variance principal direction projects every point onto the learned plane.
    """)
    return


@app.cell
def _(np, plt, torch):
    n = 1000
    _t = np.random.uniform(0, 100, size=(n, 2))
    _noise = np.random.normal(0, 1.5, size=n)
    X_np = np.stack([_t[:, 0], _t[:, 1], _t[:, 0] + _noise], axis=1)
    X = torch.tensor(X_np, dtype=torch.float32)

    x_mean = X.mean(dim=0)
    Xc = X - x_mean
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.T
    V_trunc = V[:, :-1]
    Z = Xc @ V_trunc
    Z_pad = torch.cat([Z, torch.zeros(n, 1)], dim=1)
    X_hat = Z_pad @ V.T + x_mean

    recon_err = torch.mean(torch.sum((X - X_hat) ** 2, dim=1)).item()
    print(f"Principal values (variances): {S.tolist()}")
    print(f"Mean reconstruction error:   {recon_err:.4f}")

    fig = plt.figure(figsize=(11, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=4, c="#38bdf8", alpha=0.7)
    ax1.set_title("Original 3D data")
    ax1.set_xlabel("x0"); ax1.set_ylabel("x1"); ax1.set_zlabel("x2")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(Z[:, 0], Z[:, 1], s=4, c="#a78bfa", alpha=0.7)
    ax2.set_title("2D latent code  Z = X · V[:, :-1]")
    ax2.set_xlabel("z0"); ax2.set_ylabel("z1"); ax2.set_aspect("equal")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(X_hat[:, 0], X_hat[:, 1], X_hat[:, 2], s=4, c="#f97316", alpha=0.7)
    ax3.set_title("Reconstructed 3D data")
    ax3.set_xlabel("x0"); ax3.set_ylabel("x1"); ax3.set_zlabel("x2")

    fig.tight_layout()
    fig
    return (X, X_hat, Z, x_mean, V, Vh)


# =============================================================================
# 2. S-curve: where PCA fails
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 2. When PCA fails — curved manifolds

    PCA projects onto a **hyperplane**. When the data lies on a **curved** manifold (an S-curve, a Swiss roll, a
    sphere), no plane fits well. The reconstruction error stays large even if we keep 2 of the 3 principal directions.
    We need a **nonlinear** projection — an autoencoder.
    """)
    return


@app.cell
def _(np, plt, torch):
    from sklearn.datasets import make_s_curve

    Xs_np, t = make_s_curve(n_samples=1000, noise=0.05, random_state=0)
    Xs = torch.tensor(Xs_np, dtype=torch.float32)
    Xs_c = Xs - Xs.mean(dim=0)

    _U, _S, _Vh = torch.linalg.svd(Xs_c, full_matrices=False)
    _V = _Vh.T
    Zs = Xs_c @ _V[:, :2]
    Xs_hat = Zs @ _V[:, :2].T + Xs.mean(dim=0)
    recon_s = torch.mean(torch.sum((Xs - Xs_hat) ** 2, dim=1)).item()

    fig2 = plt.figure(figsize=(10, 4))
    ax1 = fig2.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], c=t, cmap="viridis", s=6)
    ax1.set_title("S-curve in 3D")

    ax2 = fig2.add_subplot(1, 2, 2)
    ax2.scatter(Zs[:, 0], Zs[:, 1], c=t, cmap="viridis", s=6)
    ax2.set_title(f"PCA 2D projection\nMean recon err = {recon_s:.3f}")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    fig2.tight_layout()
    print(f"S-curve PCA reconstruction error: {recon_s:.4f}  (non-zero — PCA cannot unroll a curved surface)")
    fig2
    return


# =============================================================================
# 3. MNIST autoencoder
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Autoencoder on MNIST

    We train a small convolutional autoencoder with a bottleneck of size `n_z` (from the slider).
    Architecture (Chapter 14, Listings 14.2 & 14.3):

    ```
    Encoder:  Conv → BN → ReLU → MaxPool  (×3)   →   Flatten   →   Linear( · , n_z )
    Decoder:  Linear(n_z, · )   →   Unflatten   →   ConvTranspose → BN → ReLU   (×3)   →   Sigmoid
    ```
    """)
    return


@app.cell
def _(torch):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    _tx = transforms.ToTensor()
    _root = "./mnist_data"
    try:
        train_ds = datasets.MNIST(_root, train=True, download=True, transform=_tx)
        test_ds  = datasets.MNIST(_root, train=False, download=True, transform=_tx)
    except Exception as err:
        print(f"MNIST download failed ({err}); using a small random stand-in.")
        class _Fake(torch.utils.data.Dataset):
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
            def __getitem__(self, i):
                return torch.rand(1, 28, 28), i % 10
        train_ds = _Fake(2000)
        test_ds  = _Fake(400)

    BATCH = 128
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}, Batch: {BATCH}")
    return train_loader, test_loader


@app.cell
def _(nn, torch):
    class ConvEncoder(nn.Module):
        def __init__(self, n_z=2):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                      # 14
                nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                      # 7
                nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),                                      # 3 (floor(7/2))
                nn.Flatten(),
            )
            self.fc = nn.Linear(64 * 3 * 3, n_z)

        def forward(self, x):
            return self.fc(self.feat(x))

    class ConvDecoder(nn.Module):
        def __init__(self, n_z=2):
            super().__init__()
            self.fc = nn.Linear(n_z, 64 * 3 * 3)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, 2, 0, output_padding=0),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(16), nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            h = self.fc(z).view(-1, 64, 3, 3)
            return self.deconv(h)

    class AE(nn.Module):
        def __init__(self, n_z=2):
            super().__init__()
            self.encoder = ConvEncoder(n_z)
            self.decoder = ConvDecoder(n_z)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z
    return AE, ConvDecoder, ConvEncoder


@app.cell
def _(AE, F, epochs_slider, nz_slider, torch, train_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AE(n_z=int(nz_slider.value)).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)

    ae.train()
    for ep in range(int(epochs_slider.value)):
        running = 0.0
        for batch, (xb, _) in enumerate(train_loader):
            xb = xb.to(device)
            x_hat, _ = ae(xb)
            # pad/crop if needed to match shapes
            if x_hat.shape != xb.shape:
                xb = F.interpolate(xb, size=x_hat.shape[-2:])
            loss = F.mse_loss(x_hat, xb)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
            running += loss.item() * xb.size(0)
            if batch >= 200:                       # cap iterations for speed
                break
        print(f"[AE]  epoch {ep+1}/{epochs_slider.value}  loss={running/((batch+1)*xb.size(0)):.4f}")
    return ae, device


@app.cell
def _(ae, device, plt, test_loader, torch):
    ae.eval()
    with torch.no_grad():
        xb, yb = next(iter(test_loader))
        xb = xb.to(device)
        x_hat, z = ae(xb)
    xb = xb.cpu(); x_hat = x_hat.cpu()

    fig3, axes = plt.subplots(2, 8, figsize=(10, 2.6))
    for i in range(8):
        axes[0, i].imshow(xb[i, 0], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_title("input", fontsize=9)
    axes[1, 0].set_title("AE recon", fontsize=9)
    fig3.tight_layout()
    fig3
    return


# =============================================================================
# 4. VAE
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Variational Autoencoder

    The VAE's encoder emits **two** vectors: $\vec\mu(\vec x)$ and $\log\vec\sigma^2(\vec x)$.
    The latent code is sampled using the **reparameterization trick**:

    $$
    \vec z \;=\; \vec\mu \;+\; \vec\sigma \odot \vec\varepsilon, \qquad \vec\varepsilon\sim\mathcal{N}(\vec 0, I).
    $$

    The loss combines reconstruction (BCE) and the closed-form KL to $\mathcal{N}(\vec 0, I)$:

    $$
    \mathrm{KL} = \tfrac{1}{2}\sum_i\big( \mu_i^2 + \sigma_i^2 - 2\log\sigma_i - 1 \big).
    $$
    """)
    return


@app.cell
def _(ConvDecoder, ConvEncoder, nn, torch):
    class VAE(nn.Module):
        def __init__(self, n_z=2):
            super().__init__()
            # share Conv backbone, two heads for mu and log-var
            self.backbone = ConvEncoder(n_z=n_z).feat          # reuse conv stack
            self.fc_mu      = nn.Linear(64 * 3 * 3, n_z)
            self.fc_logvar  = nn.Linear(64 * 3 * 3, n_z)
            self.decoder    = ConvDecoder(n_z=n_z)

        def encode(self, x):
            h = self.backbone(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            return self.decoder(z), mu, log_var, z
    return (VAE,)


@app.cell
def _(F, VAE, beta_slider, epochs_slider, nz_slider, torch, train_loader):
    device_v = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(n_z=int(nz_slider.value)).to(device_v)
    opt_v = torch.optim.Adam(vae.parameters(), lr=1e-3)

    def vae_loss(x_hat, x, mu, log_var, beta=1.0):
        if x_hat.shape != x.shape:
            x = F.interpolate(x, size=x_hat.shape[-2:])
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + beta * kld, bce.item(), kld.item()

    vae.train()
    for ep in range(int(epochs_slider.value)):
        r_total, r_bce, r_kld, nb = 0.0, 0.0, 0.0, 0
        for batch, (xb, _) in enumerate(train_loader):
            xb = xb.to(device_v)
            x_hat, mu, log_var, _ = vae(xb)
            loss, bce, kld = vae_loss(x_hat, xb, mu, log_var, beta=float(beta_slider.value))
            opt_v.zero_grad(); loss.backward(); opt_v.step()
            r_total += loss.item(); r_bce += bce; r_kld += kld; nb += xb.size(0)
            if batch >= 200:
                break
        print(f"[VAE] epoch {ep+1}/{epochs_slider.value}  total={r_total/nb:.3f}  bce={r_bce/nb:.3f}  kld={r_kld/nb:.3f}")
    return device_v, vae


@app.cell
def _(device_v, plt, test_loader, torch, vae):
    vae.eval()
    with torch.no_grad():
        xb, yb = next(iter(test_loader))
        xb = xb.to(device_v)
        x_hat, mu, log_var, z = vae(xb)
    xb = xb.cpu(); x_hat = x_hat.cpu()

    fig4, axes = plt.subplots(2, 8, figsize=(10, 2.6))
    for i in range(8):
        axes[0, i].imshow(xb[i, 0], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_title("input", fontsize=9)
    axes[1, 0].set_title("VAE recon", fontsize=9)
    fig4.tight_layout()
    fig4
    return


# =============================================================================
# 5. AE vs VAE latent scatter
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Latent-space scatter (requires `n_z = 2`)

    With a 2D bottleneck we can plot every test digit's latent code and colour it by class.
    Observe:

    - **AE latent space**: spreads over a wide, often irregular range — gaps between clusters.
    - **VAE latent space**: concentrated near the origin, approximately unit variance per axis, continuous — sampling any point plausibly decodes to a digit.
    """)
    return


@app.cell
def _(ae, device, device_v, nz_slider, plt, test_loader, torch, vae):
    if int(nz_slider.value) != 2:
        print("Set n_z = 2 to see the 2D latent-space scatter.")
        _out = None
    else:
        ae.eval(); vae.eval()
        zs_ae, zs_vae, labels = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                _, z_a = ae(xb.to(device))
                _, mu, _, _ = vae(xb.to(device_v))
                zs_ae.append(z_a.cpu()); zs_vae.append(mu.cpu()); labels.append(yb)
        Z_ae  = torch.cat(zs_ae).numpy()
        Z_vae = torch.cat(zs_vae).numpy()
        Y     = torch.cat(labels).numpy()

        fig5, (axA, axV) = plt.subplots(1, 2, figsize=(11, 4.6))
        sc1 = axA.scatter(Z_ae[:, 0],  Z_ae[:, 1],  c=Y, cmap="tab10", s=6, alpha=0.7)
        axA.set_title("Autoencoder latent space")
        axA.set_xlabel("z0"); axA.set_ylabel("z1")
        sc2 = axV.scatter(Z_vae[:, 0], Z_vae[:, 1], c=Y, cmap="tab10", s=6, alpha=0.7)
        axV.set_title("VAE latent space (μ only)")
        axV.set_xlabel("μ0"); axV.set_ylabel("μ1")
        axV.set_xlim(-4, 4); axV.set_ylim(-4, 4)
        axV.axhline(0, color="#444", lw=0.5); axV.axvline(0, color="#444", lw=0.5)
        fig5.colorbar(sc2, ax=[axA, axV], label="digit class", fraction=0.025)
        _out = fig5
    _out
    return


# =============================================================================
# 6. Sample from the VAE prior
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Sampling new digits from $\mathcal{N}(\vec 0, I)$

    Because the VAE's latent distribution is regularized toward $\mathcal{N}(\vec 0, I)$, we can **sample latent codes from the prior** and decode them.
    Doing the same with an autoencoder generally produces garbage because its latent space has large empty regions.
    """)
    return


@app.cell
def _(device_v, nz_slider, plt, torch, vae):
    vae.eval()
    with torch.no_grad():
        z_samp = torch.randn(16, int(nz_slider.value), device=device_v)
        x_samp = vae.decoder(z_samp).cpu()
    fig6, axes = plt.subplots(2, 8, figsize=(10, 2.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_samp[i, 0], cmap="gray"); ax.axis("off")
    fig6.suptitle("VAE samples z ~ N(0, I) → decoder")
    fig6.tight_layout()
    fig6
    return


# =============================================================================
# 7. Closed-form KL explorer
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Closed-form KL divergence

    For diagonal Gaussians $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$ (1D):

    $$ \mathrm{KL}(q\Vert p) = \tfrac{1}{2}(\mu^2 + \sigma^2 - 2\log\sigma - 1). $$

    Slide $\mu$ and $\sigma$ to see how the KL loss behaves. It is minimized at $\mu=0, \sigma=1$.
    """)
    mu_s = mo.ui.slider(-3, 3, step=0.1, value=0.0, label="μ")
    sg_s = mo.ui.slider(0.1, 3.0, step=0.05, value=1.0, label="σ")
    mo.vstack([mu_s, sg_s])
    return mu_s, sg_s


@app.cell
def _(mu_s, np, plt, sg_s):
    mu = mu_s.value; sg = sg_s.value
    kl = 0.5 * (mu**2 + sg**2 - 2*np.log(sg) - 1)

    x = np.linspace(-5, 5, 400)
    q = (1/(sg*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sg)**2)
    p = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2)

    fig7, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(x, p, color="#38bdf8", alpha=0.35, label="p = N(0,1)")
    ax.fill_between(x, q, color="#f97316", alpha=0.35, label=f"q = N({mu:.2f},{sg:.2f}²)")
    ax.plot(x, p, color="#38bdf8"); ax.plot(x, q, color="#f97316")
    ax.set_title(f"KL(q || p) = {kl:.3f}")
    ax.legend()
    fig7.tight_layout()
    fig7
    return


# =============================================================================
# 8. Reparameterization trick sanity check
# =============================================================================
@app.cell
def _(mo):
    mo.md(r"""
    ## 8. The reparameterization trick

    We sample $\vec z$ two different ways and check that both produce the same distribution:

    - **Direct:** $\vec z \sim \mathcal{N}(\mu, \sigma^2)$ via `torch.normal(mu, sigma)` (not differentiable through $\mu, \sigma$).
    - **Reparameterized:** $\vec z = \mu + \sigma \cdot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$ (differentiable).
    """)
    return


@app.cell
def _(plt, torch):
    mu_t = torch.tensor([2.0], requires_grad=True)
    lv_t = torch.tensor([0.5], requires_grad=True)          # log-var
    sg_t = torch.exp(0.5 * lv_t)

    N = 20000
    direct  = torch.normal(mu_t.expand(N), sg_t.expand(N)).detach()
    eps     = torch.randn(N)
    reparam = (mu_t + sg_t * eps).detach()

    fig8, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(direct.numpy(),  bins=60, alpha=0.5, label="torch.normal(μ, σ)", color="#38bdf8")
    ax.hist(reparam.numpy(), bins=60, alpha=0.5, label="μ + σ·ε", color="#f97316")
    ax.set_title("Both paths sample from N(μ=2.0, σ²=e^0.5)")
    ax.legend()
    fig8.tight_layout()

    loss = (mu_t + sg_t * eps).mean()
    loss.backward()
    print(f"Grad wrt μ       : {mu_t.grad.item():.4f}  (flows only through the reparam path)")
    print(f"Grad wrt log_var : {lv_t.grad.item():.4f}")
    fig8
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    - **PCA** finds the best linear subspace; fails on curved data.
    - **Autoencoders** learn nonlinear manifolds but have no structure on the latent space.
    - **VAEs** add a KL regularizer and use the reparameterization trick to learn a **compact, smooth, generative** latent space.
    - The training objective is a lower bound (ELBO) on $\log p(\vec x)$.

    Try: change `β` (the KL weight) and re-train. Small β ≈ autoencoder (sharp recon, ragged latent). Large β ≈ purely
    prior-matching (blurry recon, very compact latent).
    """)
    return


if __name__ == "__main__":
    app.run()
