"""
IME 775: Probability Distributions — Raw vs PyTorch
====================================================
An interactive marimo notebook comparing hand-coded probability
distributions with PyTorch implementations, plus generative-AI
data generation examples.

Course: IME 775 - Mathematical Foundations of Deep Learning
Topics: Gaussian, Bernoulli, Categorical, Multinomial, Sampling, Generative Data
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # IME 775: Probability Distributions — Raw vs PyTorch

        ## Learning Objectives

        1. Implement probability distributions from scratch (NumPy)
        2. Compare with PyTorch's `torch.distributions` API
        3. Understand sampling, PDF/PMF evaluation, and log-likelihoods
        4. Use distributions to **generate synthetic data** for ML / generative AI

        ---

        ### Organization

        | Section | Topic |
        |---------|-------|
        | 1 | Gaussian (1D): Raw vs PyTorch |
        | 2 | Multivariate Gaussian: Covariance & Sampling |
        | 3 | Bernoulli & Binomial |
        | 4 | Categorical & Multinomial |
        | 5 | Generative AI: Synthetic Data Pipeline |
        | 6 | Gaussian Mixture Model (Mini-VAE Intuition) |
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import torch
    import torch.distributions as D
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse
    return np, torch, D, plt, gridspec, Ellipse


# ═══════════════════════════════════════════════════════════════
# SECTION 1: 1D GAUSSIAN
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 1. Gaussian Distribution (1D)

        $$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

        We'll implement this from scratch, then compare with `torch.distributions.Normal`.
        """
    )
    return


@app.cell
def __(mo):
    g_mu = mo.ui.slider(start=-3.0, stop=3.0, step=0.1, value=0.0,
                         label="μ (mean)", show_value=True)
    g_sigma = mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=1.0,
                            label="σ (std dev)", show_value=True)
    g_n = mo.ui.slider(start=100, stop=5000, step=100, value=1000,
                        label="N (samples)", show_value=True)
    mo.md(
        f"""
        ### Parameters

        | Parameter | Control |
        |-----------|---------|
        | Mean | {g_mu} |
        | Std Dev | {g_sigma} |
        | Samples | {g_n} |
        """
    )
    return g_mu, g_sigma, g_n


@app.cell
def __(np, torch, D, plt, g_mu, g_sigma, g_n):
    # ── Raw NumPy Implementation ──
    def gaussian_pdf_raw(x, mu, sigma):
        """Hand-coded Gaussian PDF."""
        coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)

    def gaussian_sample_raw(mu, sigma, n, seed=42):
        """Box-Muller transform for Gaussian sampling."""
        _rng = np.random.default_rng(seed)
        _u1 = _rng.uniform(0, 1, n)
        _u2 = _rng.uniform(0, 1, n)
        _z = np.sqrt(-2 * np.log(_u1)) * np.cos(2 * np.pi * _u2)
        return mu + sigma * _z

    def gaussian_log_likelihood_raw(samples, mu, sigma):
        """Hand-coded log-likelihood."""
        _n = len(samples)
        _ll = -_n/2 * np.log(2 * np.pi) - _n * np.log(sigma) \
             - np.sum((samples - mu)**2) / (2 * sigma**2)
        return _ll

    # ── PyTorch Implementation ──
    _mu = g_mu.value
    _sigma = g_sigma.value
    _n = g_n.value

    dist_pt = D.Normal(loc=torch.tensor(_mu), scale=torch.tensor(_sigma))
    _samples_pt = dist_pt.sample((int(_n),)).numpy()

    # ── Raw Implementation ──
    _samples_raw = gaussian_sample_raw(_mu, _sigma, int(_n))

    # ── Compare ──
    _x_grid = np.linspace(_mu - 4*_sigma, _mu + 4*_sigma, 300)
    _pdf_raw = gaussian_pdf_raw(_x_grid, _mu, _sigma)
    _pdf_pt = torch.exp(dist_pt.log_prob(torch.tensor(_x_grid))).numpy()

    _ll_raw = gaussian_log_likelihood_raw(_samples_raw, _mu, _sigma)
    _ll_pt = dist_pt.log_prob(torch.tensor(_samples_pt)).sum().item()

    _fig1, _axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Raw
    _axes1[0].hist(_samples_raw, bins=50, density=True, alpha=0.6, color='steelblue',
                  label=f'Samples (n={_n})')
    _axes1[0].plot(_x_grid, _pdf_raw, 'r-', lw=2, label='Raw PDF')
    _axes1[0].set_title(f'Raw NumPy (Box-Muller)\nμ̂={np.mean(_samples_raw):.3f}, σ̂={np.std(_samples_raw):.3f}')
    _axes1[0].set_xlabel('x')
    _axes1[0].set_ylabel('Density')
    _axes1[0].legend()
    _axes1[0].axvline(_mu, color='gray', ls='--', alpha=0.5)

    # Right: PyTorch
    _axes1[1].hist(_samples_pt, bins=50, density=True, alpha=0.6, color='coral',
                  label=f'Samples (n={_n})')
    _axes1[1].plot(_x_grid, _pdf_pt, 'b-', lw=2, label='PyTorch PDF')
    _axes1[1].set_title(f'PyTorch Normal\nμ̂={np.mean(_samples_pt):.3f}, σ̂={np.std(_samples_pt):.3f}')
    _axes1[1].set_xlabel('x')
    _axes1[1].set_ylabel('Density')
    _axes1[1].legend()
    _axes1[1].axvline(_mu, color='gray', ls='--', alpha=0.5)

    _fig1.suptitle(f'1D Gaussian: μ={_mu}, σ={_sigma}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig1
    return gaussian_pdf_raw, gaussian_sample_raw, gaussian_log_likelihood_raw, dist_pt


@app.cell
def __(mo, np, g_mu, g_sigma, g_n):
    _mu = g_mu.value
    _sigma = g_sigma.value
    _n = int(g_n.value)
    mo.md(
        f"""
        ### Code Comparison

        | | Raw (NumPy) | PyTorch |
        |---|---|---|
        | **PDF** | `1/(σ√2π) * exp(-(x-μ)²/2σ²)` | `dist.log_prob(x).exp()` |
        | **Sample** | Box-Muller transform | `dist.sample((n,))` |
        | **Log-Lik** | `Σ log p(xᵢ)` manual | `dist.log_prob(x).sum()` |

        **Takeaway:** PyTorch handles the math automatically. For production ML, always
        use `torch.distributions` — it supports automatic differentiation (gradients flow
        through sampling via the reparameterization trick).
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 2: MULTIVARIATE GAUSSIAN
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 2. Multivariate Gaussian

        $$\mathcal{N}(\vec{x} \mid \vec{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\vec{x}-\vec{\mu})^T \Sigma^{-1} (\vec{x}-\vec{\mu})\right)$$

        The covariance matrix $\Sigma$ controls the **shape and orientation** of the distribution.
        """
    )
    return


@app.cell
def __(mo):
    mv_var1 = mo.ui.slider(start=0.5, stop=5.0, step=0.25, value=2.0,
                            label="σ₁² (variance in x₁)", show_value=True)
    mv_var2 = mo.ui.slider(start=0.5, stop=5.0, step=0.25, value=1.0,
                            label="σ₂² (variance in x₂)", show_value=True)
    mv_rho = mo.ui.slider(start=-0.95, stop=0.95, step=0.05, value=0.5,
                           label="ρ (correlation)", show_value=True)
    mv_n = mo.ui.slider(start=100, stop=3000, step=100, value=500,
                         label="N (samples)", show_value=True)
    mo.md(
        f"""
        ### Parameters

        | Parameter | Control |
        |-----------|---------|
        | Variance x₁ | {mv_var1} |
        | Variance x₂ | {mv_var2} |
        | Correlation | {mv_rho} |
        | Samples | {mv_n} |
        """
    )
    return mv_var1, mv_var2, mv_rho, mv_n


@app.cell
def __(np, torch, D, plt, Ellipse, mv_var1, mv_var2, mv_rho, mv_n):
    _v1 = mv_var1.value
    _v2 = mv_var2.value
    _rho = mv_rho.value
    _n = int(mv_n.value)

    # Build covariance matrix
    _cov12 = _rho * np.sqrt(_v1 * _v2)
    _cov_np = np.array([[_v1, _cov12], [_cov12, _v2]])
    _mu_np = np.array([0.0, 0.0])

    # ── Raw NumPy: Cholesky sampling ──
    _L = np.linalg.cholesky(_cov_np)
    _rng2 = np.random.default_rng(42)
    _z = _rng2.standard_normal((_n, 2))
    _samples_raw_mv = (_z @ _L.T) + _mu_np

    # ── PyTorch ──
    _mu_pt = torch.zeros(2)
    _cov_pt = torch.tensor(_cov_np, dtype=torch.float32)
    _dist_mv = D.MultivariateNormal(_mu_pt, _cov_pt)
    _samples_pt_mv = _dist_mv.sample((int(_n),)).numpy()

    # ── Eigenvectors for ellipse ──
    _eigvals, _eigvecs = np.linalg.eigh(_cov_np)
    _angle = np.degrees(np.arctan2(_eigvecs[1, 1], _eigvecs[0, 1]))

    _fig2, _axes2 = plt.subplots(1, 2, figsize=(14, 6))

    for _idx, (_samp, _title, _color) in enumerate([
        (_samples_raw_mv, 'Raw NumPy (Cholesky)', 'steelblue'),
        (_samples_pt_mv, 'PyTorch MultivariateNormal', 'coral')
    ]):
        _ax = _axes2[_idx]
        _ax.scatter(_samp[:, 0], _samp[:, 1], alpha=0.3, s=8, c=_color)

        # Draw 1σ and 2σ ellipses
        for _k, _ls in [(1, '-'), (2, '--')]:
            _ell = Ellipse(xy=(0, 0),
                          width=2*_k*np.sqrt(_eigvals[1]),
                          height=2*_k*np.sqrt(_eigvals[0]),
                          angle=_angle, fill=False,
                          color='red', lw=2, ls=_ls,
                          label=f'{_k}σ ellipse')
            _ax.add_patch(_ell)

        # Draw eigenvectors
        for _j in range(2):
            _scale = np.sqrt(_eigvals[_j])
            _ax.annotate('', xy=_eigvecs[:, _j]*_scale*1.5,
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
            _ax.text(_eigvecs[0, _j]*_scale*1.7, _eigvecs[1, _j]*_scale*1.7,
                    f'λ={_eigvals[_j]:.2f}', fontsize=9, color='darkgreen',
                    fontweight='bold')

        _ax.set_title(f'{_title}\nρ={_rho:.2f}')
        _ax.set_xlabel('x₁')
        _ax.set_ylabel('x₂')
        _ax.set_aspect('equal')
        _ax.legend(fontsize=8)
        _lim = max(np.sqrt(_v1), np.sqrt(_v2)) * 3.5
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.grid(True, alpha=0.3)

    _fig2.suptitle('2D Gaussian: Raw Cholesky vs PyTorch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig2
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Raw vs PyTorch: Multivariate Gaussian

        ```python
        # ── RAW (NumPy) ──
        L = np.linalg.cholesky(Σ)          # Cholesky decomposition
        z = rng.standard_normal((n, d))     # Standard normal samples
        samples = z @ L.T + μ              # Transform to target distribution

        # ── PYTORCH ──
        dist = D.MultivariateNormal(μ, Σ)
        samples = dist.sample((n,))         # One line!
        log_p = dist.log_prob(samples)      # Differentiable log-probability
        ```

        **Key insight:** PyTorch's version supports **backpropagation through sampling**
        via the reparameterization trick — essential for VAEs and generative models.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 3: BERNOULLI & BINOMIAL
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 3. Bernoulli & Binomial Distributions

        **Bernoulli:** Single binary trial  →  $P(X=1) = \theta$

        **Binomial:** $n$ independent Bernoulli trials  →  $P(k) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$
        """
    )
    return


@app.cell
def __(mo):
    b_theta = mo.ui.slider(start=0.05, stop=0.95, step=0.05, value=0.6,
                            label="θ (success prob)", show_value=True)
    b_n_trials = mo.ui.slider(start=1, stop=50, step=1, value=20,
                               label="n (trials)", show_value=True)
    b_n_exp = mo.ui.slider(start=100, stop=5000, step=100, value=1000,
                            label="N (experiments)", show_value=True)
    mo.md(
        f"""
        ### Parameters

        | Parameter | Control |
        |-----------|---------|
        | θ (success) | {b_theta} |
        | n (trials) | {b_n_trials} |
        | N (experiments) | {b_n_exp} |
        """
    )
    return b_theta, b_n_trials, b_n_exp


@app.cell
def __(np, torch, D, plt, b_theta, b_n_trials, b_n_exp):
    _theta = b_theta.value
    _n_t = int(b_n_trials.value)
    _n_e = int(b_n_exp.value)

    # ── Raw Bernoulli ──
    _rng3 = np.random.default_rng(42)
    _bern_raw = (_rng3.uniform(0, 1, _n_e) < _theta).astype(int)

    # ── PyTorch Bernoulli ──
    _bern_pt = D.Bernoulli(probs=torch.tensor(_theta))
    _bern_samples_pt = _bern_pt.sample((int(_n_e),)).numpy().astype(int)

    # ── Raw Binomial (sum of Bernoulli) ──
    _binom_raw = np.array([
        np.sum(_rng3.uniform(0, 1, _n_t) < _theta) for _ in range(_n_e)
    ])

    # ── PyTorch Binomial ──
    _binom_pt = D.Binomial(total_count=_n_t, probs=torch.tensor(_theta))
    _binom_samples_pt = _binom_pt.sample((int(_n_e),)).numpy().astype(int)

    # ── Raw Binomial PMF ──
    from math import comb as _comb
    _k_vals = np.arange(0, _n_t + 1)
    _pmf_raw = np.array([_comb(_n_t, int(_kk)) * _theta**_kk * (1-_theta)**(_n_t-_kk)
                         for _kk in _k_vals])

    _fig3, _axes3 = plt.subplots(1, 3, figsize=(16, 5))

    # Bernoulli comparison
    for _i, (_samp, _title, _color) in enumerate([
        (_bern_raw, 'Raw NumPy', 'steelblue'),
        (_bern_samples_pt, 'PyTorch', 'coral')
    ]):
        _counts = [np.sum(_samp == 0), np.sum(_samp == 1)]
        _axes3[0].bar([_i*0.35, _i*0.35 + 0.15],
                     [_counts[0]/_n_e, _counts[1]/_n_e],
                     width=0.12, label=_title, color=_color, alpha=0.7)
    _axes3[0].set_xticks([0.075, 0.425])
    _axes3[0].set_xticklabels(['NumPy\n0    1', 'PyTorch\n0    1'])
    _axes3[0].set_title(f'Bernoulli(θ={_theta})')
    _axes3[0].set_ylabel('Frequency')
    _axes3[0].axhline(1-_theta, color='gray', ls='--', alpha=0.5, label=f'True P(0)={1-_theta:.2f}')
    _axes3[0].axhline(_theta, color='gray', ls=':', alpha=0.5, label=f'True P(1)={_theta:.2f}')
    _axes3[0].legend(fontsize=7)

    # Binomial: Raw
    _axes3[1].hist(_binom_raw, bins=np.arange(-0.5, _n_t + 1.5), density=True,
                  alpha=0.6, color='steelblue', label='Raw samples')
    _axes3[1].plot(_k_vals, _pmf_raw, 'ro-', ms=4, lw=1.5, label='Raw PMF')
    _axes3[1].set_title(f'Binomial Raw (n={_n_t}, θ={_theta})\nE={_n_t*_theta:.1f}')
    _axes3[1].set_xlabel('k (successes)')
    _axes3[1].legend(fontsize=8)

    # Binomial: PyTorch
    _axes3[2].hist(_binom_samples_pt, bins=np.arange(-0.5, _n_t + 1.5), density=True,
                  alpha=0.6, color='coral', label='PyTorch samples')
    _pmf_pt = torch.exp(_binom_pt.log_prob(torch.tensor(_k_vals, dtype=torch.float32))).numpy()
    _axes3[2].plot(_k_vals, _pmf_pt, 'b^-', ms=4, lw=1.5, label='PyTorch PMF')
    _axes3[2].set_title(f'Binomial PyTorch (n={_n_t}, θ={_theta})\nE={_n_t*_theta:.1f}')
    _axes3[2].set_xlabel('k (successes)')
    _axes3[2].legend(fontsize=8)

    _fig3.suptitle('Bernoulli & Binomial: Raw vs PyTorch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig3
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 4: CATEGORICAL & MULTINOMIAL
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 4. Categorical & Multinomial

        **Categorical:** Single draw from $K$ classes  →  $P(X=k) = \theta_k$

        **Multinomial:** $n$ draws from $K$ classes  →  counts $\vec{m}$, $\sum m_k = n$

        These are the distributions behind **softmax classifiers** and **bag-of-words** models.
        """
    )
    return


@app.cell
def __(mo):
    cat_probs_input = mo.ui.text(
        value="0.5, 0.3, 0.15, 0.05",
        label="Class probabilities (comma-separated)",
    )
    cat_labels_input = mo.ui.text(
        value="cat, dog, bird, fish",
        label="Class labels (comma-separated)",
    )
    cat_n_draws = mo.ui.slider(start=10, stop=1000, step=10, value=200,
                                label="N (draws)", show_value=True)
    mo.md(
        f"""
        ### Parameters

        | Parameter | Control |
        |-----------|---------|
        | Probabilities | {cat_probs_input} |
        | Labels | {cat_labels_input} |
        | N draws | {cat_n_draws} |

        **Try:** Change probabilities (must sum to 1) and see how counts change!
        """
    )
    return cat_probs_input, cat_labels_input, cat_n_draws


@app.cell
def __(np, torch, D, plt, cat_probs_input, cat_labels_input, cat_n_draws):
    # Parse inputs
    _probs = np.array([float(x.strip()) for x in cat_probs_input.value.split(',')])
    _probs = _probs / _probs.sum()  # normalize
    _labels = [x.strip() for x in cat_labels_input.value.split(',')]
    _K = len(_probs)
    _n_d = int(cat_n_draws.value)

    # ── Raw Categorical: inverse CDF method ──
    _rng4 = np.random.default_rng(42)
    _cdf = np.cumsum(_probs)
    _u = _rng4.uniform(0, 1, _n_d)
    _cat_raw = np.searchsorted(_cdf, _u)

    # ── PyTorch Categorical ──
    _cat_dist = D.Categorical(probs=torch.tensor(_probs, dtype=torch.float32))
    _cat_pt = _cat_dist.sample((int(_n_d),)).numpy()

    # ── Raw Multinomial: count from categorical draws ──
    _multi_raw = np.bincount(_cat_raw, minlength=_K)

    # ── PyTorch Multinomial ──
    _multi_dist = D.Multinomial(total_count=_n_d,
                                probs=torch.tensor(_probs, dtype=torch.float32))
    _multi_pt = _multi_dist.sample().numpy().astype(int)

    _fig4, _axes4 = plt.subplots(1, 3, figsize=(16, 5))

    # Categorical comparison
    _raw_freq = np.bincount(_cat_raw, minlength=_K) / _n_d
    _pt_freq = np.bincount(_cat_pt, minlength=_K) / _n_d
    _x = np.arange(_K)
    _w = 0.25
    _axes4[0].bar(_x - _w, _probs, _w, label='True θ', color='gray', alpha=0.7)
    _axes4[0].bar(_x, _raw_freq, _w, label='Raw freq', color='steelblue', alpha=0.7)
    _axes4[0].bar(_x + _w, _pt_freq, _w, label='PyTorch freq', color='coral', alpha=0.7)
    _axes4[0].set_xticks(_x)
    _axes4[0].set_xticklabels(_labels[:_K], rotation=20)
    _axes4[0].set_title(f'Categorical (n={_n_d})')
    _axes4[0].set_ylabel('Frequency')
    _axes4[0].legend(fontsize=8)

    # Multinomial Raw
    _axes4[1].bar(_x, _multi_raw, color='steelblue', alpha=0.7, label='Raw counts')
    _axes4[1].bar(_x, _n_d * _probs, color='none', edgecolor='red', lw=2,
                 label='Expected', ls='--')
    _axes4[1].set_xticks(_x)
    _axes4[1].set_xticklabels(_labels[:_K], rotation=20)
    _axes4[1].set_title(f'Multinomial Raw (n={_n_d})')
    _axes4[1].set_ylabel('Count')
    _axes4[1].legend(fontsize=8)

    # Multinomial PyTorch
    _axes4[2].bar(_x, _multi_pt, color='coral', alpha=0.7, label='PyTorch counts')
    _axes4[2].bar(_x, _n_d * _probs, color='none', edgecolor='blue', lw=2,
                 label='Expected', ls='--')
    _axes4[2].set_xticks(_x)
    _axes4[2].set_xticklabels(_labels[:_K], rotation=20)
    _axes4[2].set_title(f'Multinomial PyTorch (n={_n_d})')
    _axes4[2].set_ylabel('Count')
    _axes4[2].legend(fontsize=8)

    _fig4.suptitle('Categorical & Multinomial: Raw vs PyTorch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig4
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Softmax Connection

        In neural networks, a **softmax** layer converts logits to a Categorical distribution:

        ```python
        logits = model(x)                          # raw scores, shape (K,)
        probs = torch.softmax(logits, dim=-1)       # → Categorical parameters
        dist = D.Categorical(probs=probs)
        predicted_class = dist.sample()              # or probs.argmax()
        loss = -dist.log_prob(true_label)            # cross-entropy loss!
        ```

        This is exactly how **multi-class classification** works.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 5: GENERATIVE AI — SYNTHETIC DATA PIPELINE
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 5. Generative AI: Synthetic Data for ML

        Real generative models (GANs, VAEs, diffusion) learn to **sample from complex
        distributions**. Here we'll build the intuition step-by-step:

        1. **Class-conditional generation:** Different Gaussians per class
        2. **Synthetic tabular data:** Multi-feature data with correlations
        3. **Text generation (bag-of-words):** Multinomial sampling of words

        All of these are building blocks used in production generative AI systems.
        """
    )
    return


@app.cell
def __(mo):
    gen_n_per_class = mo.ui.slider(start=50, stop=500, step=50, value=200,
                                    label="Samples per class", show_value=True)
    gen_separation = mo.ui.slider(start=0.5, stop=5.0, step=0.25, value=2.0,
                                   label="Class separation", show_value=True)
    mo.md(
        f"""
        ### 5a. Class-Conditional Generation

        Generate synthetic classification data: each class has its own Gaussian.

        | Parameter | Control |
        |-----------|---------|
        | Samples/class | {gen_n_per_class} |
        | Class separation | {gen_separation} |
        """
    )
    return gen_n_per_class, gen_separation


@app.cell
def __(np, torch, D, plt, gen_n_per_class, gen_separation):
    _n_pc = int(gen_n_per_class.value)
    _sep = gen_separation.value

    # Define 3 class-conditional Gaussians
    _class_params = [
        {'mu': torch.tensor([0.0, 0.0]),
         'cov': torch.tensor([[1.0, 0.3], [0.3, 0.8]]),
         'label': 'Class A', 'color': '#2196F3'},
        {'mu': torch.tensor([_sep, _sep]),
         'cov': torch.tensor([[0.8, -0.2], [-0.2, 1.2]]),
         'label': 'Class B', 'color': '#FF5722'},
        {'mu': torch.tensor([-_sep, _sep]),
         'cov': torch.tensor([[1.5, 0.0], [0.0, 0.5]]),
         'label': 'Class C', 'color': '#4CAF50'},
    ]

    _fig5, _axes5 = plt.subplots(1, 2, figsize=(14, 6))

    all_data = []
    all_labels = []

    for _i, _cp in enumerate(_class_params):
        _dist_cc = D.MultivariateNormal(_cp['mu'], _cp['cov'])
        _samples_cc = _dist_cc.sample((_n_pc,)).numpy()
        all_data.append(_samples_cc)
        all_labels.extend([_i] * _n_pc)

        _axes5[0].scatter(_samples_cc[:, 0], _samples_cc[:, 1],
                         alpha=0.4, s=15, c=_cp['color'], label=_cp['label'])

    _axes5[0].set_title('Generated Training Data\n(Class-Conditional Gaussians)')
    _axes5[0].set_xlabel('Feature 1')
    _axes5[0].set_ylabel('Feature 2')
    _axes5[0].legend()
    _axes5[0].grid(True, alpha=0.3)
    _axes5[0].set_aspect('equal')

    # Show per-class statistics
    _all = np.vstack(all_data)
    _labs = np.array(all_labels)
    for _i, _cp in enumerate(_class_params):
        _mask = _labs == _i
        _subset = _all[_mask]
        _mu_hat = _subset.mean(axis=0)
        _axes5[1].bar(_i, np.linalg.det(np.cov(_subset.T)), color=_cp['color'],
                     alpha=0.7, label=f'{_cp["label"]}: μ̂=({_mu_hat[0]:.1f},{_mu_hat[1]:.1f})')
    _axes5[1].set_xticks([0, 1, 2])
    _axes5[1].set_xticklabels(['A', 'B', 'C'])
    _axes5[1].set_ylabel('det(Σ̂) — Generalized Variance')
    _axes5[1].set_title('Estimated Covariance "Volume"')
    _axes5[1].legend(fontsize=8)

    _fig5.suptitle('Synthetic Data Generation for Classification',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig5
    return all_data, all_labels


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 5b. Synthetic Text Generation (Bag of Words)

        In NLP, a simple generative model treats documents as draws from a Multinomial
        distribution over vocabulary:

        $$P(\text{document}) = \text{Multinomial}(\vec{m} \mid n, \vec{\theta}_{\text{topic}})$$

        This is the foundation of **Naive Bayes classifiers** and **topic models**.
        """
    )
    return


@app.cell
def __(np, torch, D, plt):
    # Define topic-word distributions
    vocab = ['neural', 'network', 'gradient', 'loss', 'train',
             'goal', 'score', 'team', 'player', 'win',
             'stock', 'market', 'profit', 'trade', 'risk']

    topic_probs = {
        'ML': torch.tensor([0.18, 0.15, 0.14, 0.13, 0.12,
                            0.01, 0.01, 0.01, 0.01, 0.01,
                            0.05, 0.05, 0.05, 0.04, 0.04]),
        'Sports': torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01,
                                0.18, 0.16, 0.15, 0.14, 0.13,
                                0.04, 0.04, 0.04, 0.04, 0.03]),
        'Finance': torch.tensor([0.03, 0.03, 0.03, 0.03, 0.03,
                                 0.03, 0.03, 0.03, 0.03, 0.03,
                                 0.16, 0.15, 0.13, 0.12, 0.14]),
    }

    # Generate synthetic documents
    _doc_length = 50
    _n_docs_per_topic = 5
    _topic_colors = {'ML': '#2196F3', 'Sports': '#FF5722', 'Finance': '#4CAF50'}

    _fig6, _axes6 = plt.subplots(1, 3, figsize=(16, 5))

    for _idx, (_topic, _probs) in enumerate(topic_probs.items()):
        _multi = D.Multinomial(total_count=_doc_length, probs=_probs)

        # Generate several documents
        _docs = _multi.sample((_n_docs_per_topic,)).numpy()
        _avg_counts = _docs.mean(axis=0)

        _axes6[_idx].barh(range(len(vocab)), _avg_counts,
                        color=_topic_colors[_topic], alpha=0.7)
        _axes6[_idx].set_yticks(range(len(vocab)))
        _axes6[_idx].set_yticklabels(vocab, fontsize=8)
        _axes6[_idx].set_xlabel('Avg word count')
        _axes6[_idx].set_title(f'Topic: {_topic}\n({_n_docs_per_topic} docs × {_doc_length} words)')
        _axes6[_idx].invert_yaxis()

    _fig6.suptitle('Synthetic Document Generation via Multinomial Sampling',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig6
    return vocab, topic_probs


@app.cell
def __(mo, torch, D, vocab, topic_probs):
    # Show example generated document
    _topic = 'ML'
    _multi = D.Multinomial(total_count=30, probs=topic_probs[_topic])
    _doc_counts = _multi.sample().int()
    _words = []
    for _i, _count in enumerate(_doc_counts.tolist()):
        _words.extend([vocab[_i]] * _count)

    import random as _random
    _random.seed(42)
    _random.shuffle(_words)
    _doc_text = ' '.join(_words)

    mo.md(
        f"""
        ### Example Generated ML Document (30 words)

        > {_doc_text}

        **How it works:**
        1. Choose topic → selects $\\vec{{\\theta}}_{{\\text{{topic}}}}$
        2. Sample word counts from $\\text{{Multinomial}}(n=30, \\vec{{\\theta}})$
        3. Shuffle words → synthetic document

        This is simplified, but the **same principle** underlies:
        - **LDA** (Latent Dirichlet Allocation) for topic modeling
        - **Naive Bayes** for text classification
        - **Language model pretraining** (predicting next token = sampling from Categorical)
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 6: GAUSSIAN MIXTURE MODEL (VAE INTUITION)
# ═══════════════════════════════════════════════════════════════

@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## 6. Gaussian Mixture Model — The Road to VAEs

        A **Gaussian Mixture Model (GMM)** is a generative model:

        $$p(\vec{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\vec{x} \mid \vec{\mu}_k, \Sigma_k)$$

        **Connection to generative AI:**
        - GMM: discrete latent variable $z \in \{1,\ldots,K\}$ → Gaussian per component
        - **VAE**: continuous latent variable $\vec{z} \sim \mathcal{N}(\vec{0}, I)$ → neural network decoder

        Both follow the pattern: **sample latent → decode to data space**.
        """
    )
    return


@app.cell
def __(mo):
    gmm_k = mo.ui.slider(start=2, stop=6, step=1, value=3,
                          label="K (components)", show_value=True)
    gmm_n = mo.ui.slider(start=200, stop=3000, step=100, value=1000,
                          label="N (total samples)", show_value=True)
    mo.md(
        f"""
        ### Parameters

        | Parameter | Control |
        |-----------|---------|
        | K (components) | {gmm_k} |
        | N (samples) | {gmm_n} |
        """
    )
    return gmm_k, gmm_n


@app.cell
def __(np, torch, D, plt, Ellipse, gmm_k, gmm_n):
    _K = int(gmm_k.value)
    _N = int(gmm_n.value)
    torch.manual_seed(42)

    # Create random GMM parameters
    _colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']
    _mus = []
    _covs = []
    _weights = torch.ones(_K) / _K

    _rng5 = np.random.default_rng(42)
    for _k in range(_K):
        _angle = 2 * np.pi * _k / _K
        _r = 3.0
        _mu_k = torch.tensor([_r * np.cos(_angle), _r * np.sin(_angle)],
                              dtype=torch.float32)
        _mus.append(_mu_k)
        # Random covariance
        _A = torch.randn(2, 2) * 0.5
        _cov_k = _A @ _A.T + 0.3 * torch.eye(2)
        _covs.append(_cov_k)

    # ── Step 1: Sample component assignments (Categorical) ──
    _comp_dist = D.Categorical(probs=_weights)
    _assignments = _comp_dist.sample((_N,))

    # ── Step 2: Sample from corresponding Gaussian ──
    _all_samples = torch.zeros(_N, 2)
    for _k in range(_K):
        _mask = (_assignments == _k)
        _n_k = _mask.sum().item()
        if _n_k > 0:
            _gauss_k = D.MultivariateNormal(_mus[_k], _covs[_k])
            _all_samples[_mask] = _gauss_k.sample((int(_n_k),))

    _samples_np = _all_samples.numpy()
    _assign_np = _assignments.numpy()

    _fig7, _axes7 = plt.subplots(1, 3, figsize=(18, 5.5))

    # Left: Color-coded by component
    for _k in range(_K):
        _mask = _assign_np == _k
        _axes7[0].scatter(_samples_np[_mask, 0], _samples_np[_mask, 1],
                         s=10, alpha=0.4, c=_colors[_k], label=f'k={_k+1}')
    _axes7[0].set_title('GMM Samples\n(colored by component)')
    _axes7[0].legend(fontsize=8)
    _axes7[0].set_aspect('equal')
    _axes7[0].grid(True, alpha=0.3)

    # Middle: What we observe (no labels)
    _axes7[1].scatter(_samples_np[:, 0], _samples_np[:, 1],
                     s=10, alpha=0.4, c='gray')
    # Draw component ellipses
    for _k in range(_K):
        _ev, _evec = np.linalg.eigh(_covs[_k].numpy())
        _ang = np.degrees(np.arctan2(_evec[1, 1], _evec[0, 1]))
        for _sigma_mult in [1, 2]:
            _ell = Ellipse(xy=_mus[_k].numpy(),
                           width=2*_sigma_mult*np.sqrt(_ev[1]),
                           height=2*_sigma_mult*np.sqrt(_ev[0]),
                           angle=_ang, fill=False,
                           color=_colors[_k], lw=2 if _sigma_mult==1 else 1,
                           ls='-' if _sigma_mult==1 else '--')
            _axes7[1].add_patch(_ell)
        _axes7[1].plot(*_mus[_k].numpy(), 'x', color=_colors[_k], ms=12, mew=3)
    _axes7[1].set_title('Observed Data + True Components\n(latent z is hidden)')
    _axes7[1].set_aspect('equal')
    _axes7[1].grid(True, alpha=0.3)

    # Right: Generative process diagram
    _axes7[2].axis('off')
    _txt = (
        "Generative Process:\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "GMM (This notebook):\n"
        "  z ~ Categorical(π)     ← pick component\n"
        "  x ~ N(μ_z, Σ_z)       ← sample from it\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "VAE (Chapter 14):\n"
        "  z ~ N(0, I)            ← sample latent\n"
        "  x = Decoder(z)         ← neural net maps\n"
        "                           z → data space\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Diffusion (DALL·E, etc.):\n"
        "  x_T ~ N(0, I)          ← pure noise\n"
        "  x_{t-1} = Denoise(x_t) ← iterative\n"
        "  ...                       refinement\n"
        "  x_0 = final image\n"
    )
    _axes7[2].text(0.05, 0.95, _txt, transform=_axes7[2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    _axes7[2].set_title('From GMM to Modern Generative AI')

    _fig7.suptitle(f'Gaussian Mixture Model (K={_K}, N={_N})',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig7
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Summary: Raw vs PyTorch

        | Distribution | Raw Implementation | PyTorch |
        |---|---|---|
        | **Gaussian 1D** | Box-Muller transform | `D.Normal(μ, σ)` |
        | **Gaussian nD** | Cholesky decomposition | `D.MultivariateNormal(μ, Σ)` |
        | **Bernoulli** | Threshold uniform sample | `D.Bernoulli(θ)` |
        | **Binomial** | Sum of Bernoulli trials | `D.Binomial(n, θ)` |
        | **Categorical** | Inverse CDF (searchsorted) | `D.Categorical(θ)` |
        | **Multinomial** | Count categorical draws | `D.Multinomial(n, θ)` |

        ### Why PyTorch Wins for ML

        1. **Automatic differentiation** — gradients flow through `log_prob()` and `rsample()`
        2. **Reparameterization trick** — enables training VAEs and other latent variable models
        3. **GPU support** — batch sampling on CUDA
        4. **Composability** — build complex generative models from simple distributions

        ### The Generative AI Connection

        Every generative model follows the same pattern:

        **Sample latent** $\vec{z} \sim p(\vec{z})$ → **Decode** $\vec{x} = f_\theta(\vec{z})$

        - **GMM:** $z$ is discrete (Categorical), $f$ is Gaussian lookup
        - **VAE:** $z$ is continuous (Gaussian), $f$ is a neural network
        - **Diffusion:** $z$ is noise (Gaussian), $f$ is iterative denoising
        - **LLM:** $z$ is context, $f$ produces Categorical over next token
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        *IME 775 — Mathematical Foundations of Deep Learning — Week 6*
        """
    )
    return


if __name__ == "__main__":
    app.run()
