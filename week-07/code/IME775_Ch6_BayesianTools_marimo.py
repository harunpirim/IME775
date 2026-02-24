"""
IME 775: Bayesian Tools — Interactive Practice Notebook
========================================================
An interactive marimo notebook covering Bayes' theorem, entropy,
cross-entropy, KL divergence, MLE, and MAP estimation with
quiz-style practice problems and hidden solutions.

Course: IME 775 - Mathematical Foundations of Deep Learning
Chapter: 6 — Bayesian Tools
Topics: Bayes' theorem, Entropy, Cross-Entropy, KL Divergence, MLE, MAP
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # IME 775: Bayesian Tools — Chapter 6 Practice

    ## Learning Objectives

    1. Apply Bayes' theorem to classification and diagnostic problems
    2. Compute entropy, cross-entropy, and KL divergence by hand and in code
    3. Derive MLE estimates for Gaussian and Bernoulli distributions
    4. Understand MAP as regularized MLE and connect priors to L2/L1 penalties
    5. Visualize how prior strength affects MAP estimates

    ---

    ### Organization

    | Section | Topic | Type |
    |---------|-------|------|
    | 1 | Bayes' Theorem | Demo + Quiz |
    | 2 | Shannon Entropy | Demo + Quiz |
    | 3 | Cross-Entropy Loss | Demo + Quiz |
    | 4 | KL Divergence | Demo + Quiz |
    | 5 | Maximum Likelihood Estimation | Demo + Quiz |
    | 6 | MAP Estimation | Demo + Quiz |
    | 7 | Putting It All Together | Summary |
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import torch.distributions as D
    import matplotlib.pyplot as plt
    from scipy import stats

    return D, np, plt, stats, torch


# ═══════════════════════════════════════════════════════════════
# SECTION 1: BAYES' THEOREM
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Bayes' Theorem

    $$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}$$

    | Term | Name | Role |
    |------|------|------|
    | $P(\theta \mid \mathcal{D})$ | Posterior | Updated belief after data |
    | $P(\mathcal{D} \mid \theta)$ | Likelihood | How probable data is under $\theta$ |
    | $P(\theta)$ | Prior | Initial belief before data |
    | $P(\mathcal{D})$ | Evidence | Normalizing constant |

    Adjust the sliders below to see how prior probability and likelihood interact.
    """)
    return


@app.cell
def _(mo):
    b_prior = mo.ui.slider(start=0.01, stop=0.50, step=0.01, value=0.01,
                            label="P(disease) — prior/base rate", show_value=True)
    b_sensitivity = mo.ui.slider(start=0.50, stop=1.00, step=0.01, value=0.95,
                                  label="Sensitivity P(+|disease)", show_value=True)
    b_specificity = mo.ui.slider(start=0.50, stop=1.00, step=0.01, value=0.90,
                                  label="Specificity P(-|no disease)", show_value=True)
    mo.md(
        f"""
        ### Medical Diagnosis Example

        | Parameter | Control |
        |-----------|---------|
        | Base rate | {b_prior} |
        | Sensitivity | {b_sensitivity} |
        | Specificity | {b_specificity} |
        """
    )
    return b_prior, b_sensitivity, b_specificity


@app.cell
def _(b_prior, b_sensitivity, b_specificity, np, plt):
    _prior = b_prior.value
    _sens = b_sensitivity.value
    _spec = b_specificity.value
    _fpr = 1 - _spec

    # Bayes calculation
    _p_pos = _sens * _prior + _fpr * (1 - _prior)
    _posterior = (_sens * _prior) / _p_pos

    # Visualize
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: probability flow (natural frequencies for 10000 people)
    _N = 10000
    _sick = int(_N * _prior)
    _healthy = _N - _sick
    _tp = int(_sick * _sens)
    _fp = int(_healthy * _fpr)
    _fn = _sick - _tp
    _tn = _healthy - _fp

    _labels = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    _counts = [_tp, _fp, _fn, _tn]
    _colors = ['#22c55e', '#ef4444', '#f97316', '#3b82f6']
    _bars = _ax1.bar(_labels, _counts, color=_colors, edgecolor='white', linewidth=1.5)
    _ax1.set_title(f'Natural Frequencies (per {_N:,} people)', fontweight='bold')
    _ax1.set_ylabel('Count')
    for _bar, _count in zip(_bars, _counts):
        _ax1.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + _N*0.01,
                 str(_count), ha='center', va='bottom', fontweight='bold')

    # Right: prior vs posterior
    _categories = ['Prior\nP(disease)', 'Posterior\nP(disease|+)']
    _values = [_prior, _posterior]
    _cols = ['#94a3b8', '#8b5cf6']
    _bars2 = _ax2.bar(_categories, _values, color=_cols, width=0.5, edgecolor='white', linewidth=1.5)
    _ax2.set_ylim(0, 1)
    _ax2.set_ylabel('Probability')
    _ax2.set_title('Prior → Posterior Update', fontweight='bold')
    for _bar, _val in zip(_bars2, _values):
        _ax2.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.02,
                 f'{_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
    _ax2.axhline(0.5, color='gray', ls='--', alpha=0.3)

    _fig.suptitle("Bayes' Theorem: Medical Diagnosis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(b_prior, b_sensitivity, b_specificity, mo):
    _prior = b_prior.value
    _sens = b_sensitivity.value
    _spec = b_specificity.value
    _fpr = 1 - _spec
    _p_pos = _sens * _prior + _fpr * (1 - _prior)
    _posterior = (_sens * _prior) / _p_pos
    mo.md(
        f"""
        ### Calculation Breakdown

        | Step | Formula | Value |
        |------|---------|-------|
        | P(+) evidence | sens × prior + fpr × (1-prior) | {_sens:.2f} × {_prior:.3f} + {_fpr:.2f} × {1-_prior:.3f} = **{_p_pos:.4f}** |
        | P(disease ∣ +) | (sens × prior) / P(+) | ({_sens:.2f} × {_prior:.3f}) / {_p_pos:.4f} = **{_posterior:.4f}** |

        **Interpretation:** Despite {_sens*100:.0f}% sensitivity, a positive test only gives
        {_posterior*100:.1f}% disease probability because the base rate is just {_prior*100:.1f}%.

        {"⚠️ **Base rate fallacy:** Most positive tests are false positives!" if _posterior < 0.5 else "✅ Posterior > 50% — test is informative at this base rate."}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 1: Bayes' Theorem

    **Problem:** A spam filter has: P(spam) = 0.3, P("free" | spam) = 0.8, P("free" | not spam) = 0.05.
    An email contains "free". What is P(spam | "free")?
    """)
    return


@app.cell
def _(mo):
    _q1 = mo.ui.number(start=0, stop=1, step=0.001, value=0.0,
                        label="Your answer P(spam | 'free') = ")
    mo.md(f"""
    {_q1}

    Enter your answer (to 3 decimal places), then reveal the solution below.
    """)
    return


@app.cell
def _(mo):
    _show_q1 = mo.ui.switch(label="Show Solution", value=False)
    _show_q1
    return (_show_q1,)


@app.cell
def _(_show_q1, mo):
    if _show_q1.value:
        mo.md(r"""
        **Solution:**

        - P("free") = P("free"|spam)P(spam) + P("free"|¬spam)P(¬spam)
        - P("free") = 0.8 × 0.3 + 0.05 × 0.7 = 0.24 + 0.035 = **0.275**
        - P(spam|"free") = (0.8 × 0.3) / 0.275 = 0.24 / 0.275 ≈ **0.873**

        The word "free" raises spam probability from 30% to 87.3%.
        """)
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 2: ENTROPY
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Shannon Entropy

    **Discrete:** $H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$ (bits)

    **Continuous (Gaussian):** $H(X) = \frac{1}{2}\ln(2\pi e \sigma^2)$ (nats)

    Entropy measures **uncertainty** — higher entropy = more unpredictable.
    """)
    return


@app.cell
def _(mo):
    e_p1 = mo.ui.slider(start=0.01, stop=0.98, step=0.01, value=0.5,
                          label="p₁ (binary distribution)", show_value=True)
    e_sigma = mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0,
                            label="σ (Gaussian std dev)", show_value=True)
    mo.md(
        f"""
        ### Interactive Entropy

        | Parameter | Control |
        |-----------|---------|
        | Binary p₁ | {e_p1} |
        | Gaussian σ | {e_sigma} |
        """
    )
    return e_p1, e_sigma


@app.cell
def _(e_p1, e_sigma, np, plt):
    _p1 = e_p1.value
    _p2 = 1 - _p1
    _sigma = e_sigma.value

    # Binary entropy
    _H_bin = -(_p1 * np.log2(_p1) + _p2 * np.log2(_p2)) if 0 < _p1 < 1 else 0

    # Gaussian differential entropy
    _H_gauss = 0.5 * np.log(2 * np.pi * np.e * _sigma**2)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Binary entropy curve
    _ps = np.linspace(0.01, 0.99, 200)
    _Hs = -(_ps * np.log2(_ps) + (1 - _ps) * np.log2(1 - _ps))
    _ax1.plot(_ps, _Hs, 'b-', lw=2)
    _ax1.axvline(_p1, color='red', ls='--', lw=1.5, label=f'p₁={_p1:.2f}')
    _ax1.plot(_p1, _H_bin, 'ro', markersize=10, zorder=5)
    _ax1.set_xlabel('p₁')
    _ax1.set_ylabel('H(X) bits')
    _ax1.set_title(f'Binary Entropy: H = {_H_bin:.4f} bits', fontweight='bold')
    _ax1.legend()
    _ax1.set_xlim(0, 1)
    _ax1.set_ylim(0, 1.1)
    _ax1.grid(True, alpha=0.3)

    # Right: Gaussian entropy vs sigma
    _sigmas = np.linspace(0.1, 5.0, 200)
    _Hg = 0.5 * np.log(2 * np.pi * np.e * _sigmas**2)
    _ax2.plot(_sigmas, _Hg, 'b-', lw=2)
    _ax2.axvline(_sigma, color='red', ls='--', lw=1.5, label=f'σ={_sigma:.1f}')
    _ax2.plot(_sigma, _H_gauss, 'ro', markersize=10, zorder=5)
    _ax2.set_xlabel('σ')
    _ax2.set_ylabel('H(X) nats')
    _ax2.set_title(f'Gaussian Entropy: H = {_H_gauss:.4f} nats', fontweight='bold')
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _ax2.axhline(0, color='gray', ls='-', alpha=0.3)

    _fig.suptitle('Entropy: Discrete vs Continuous', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 2: Entropy

    **Problem A:** Compute H(X) for p = (0.7, 0.2, 0.1) in bits.

    **Problem B:** A Gaussian has σ = 2. What is its differential entropy in nats?
    """)
    return


@app.cell
def _(mo):
    _q2a = mo.ui.number(start=0, stop=5, step=0.001, value=0.0,
                         label="Answer A: H(X) = ")
    _q2b = mo.ui.number(start=-5, stop=10, step=0.001, value=0.0,
                         label="Answer B: H(X) = ")
    mo.md(f"""
    {_q2a} bits

    {_q2b} nats
    """)
    return


@app.cell
def _(mo):
    _show_q2 = mo.ui.switch(label="Show Solution", value=False)
    _show_q2
    return (_show_q2,)


@app.cell
def _(_show_q2, mo, np):
    if _show_q2.value:
        _H_a = -(0.7*np.log2(0.7) + 0.2*np.log2(0.2) + 0.1*np.log2(0.1))
        _H_b = 0.5 * np.log(2 * np.pi * np.e * 4)
        mo.md(
            f"""
            **Solution A:**
            H = -(0.7 log₂ 0.7 + 0.2 log₂ 0.2 + 0.1 log₂ 0.1)
            = -(0.7×(-0.515) + 0.2×(-2.322) + 0.1×(-3.322))
            = 0.360 + 0.464 + 0.332 = **{_H_a:.4f} bits**

            **Solution B:**
            H = ½ ln(2πe × σ²) = ½ ln(2πe × 4) = ½ ln({2*np.pi*np.e*4:.3f}) = **{_H_b:.4f} nats**

            Note: differential entropy can be negative (when σ < 1/√(2πe) ≈ 0.242).
            """
        )
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 3: CROSS-ENTROPY
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Cross-Entropy Loss

    **Discrete:** $H(p, q) = -\sum_i p(x_i) \log q(x_i)$

    **Classification loss (one-hot p):** $\mathcal{L} = -\log q(\text{true class})$

    **Gaussian:** $H(p, q) = \frac{1}{2}\ln(2\pi\sigma_2^2) + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2}$

    Below, adjust model logits for a 3-class problem and watch the loss change.
    """)
    return


@app.cell
def _(mo):
    ce_true = mo.ui.dropdown(options={"Cat (class 0)": 0, "Dog (class 1)": 1, "Bird (class 2)": 2},
                              value="Cat (class 0)", label="True class")
    ce_l0 = mo.ui.slider(start=-5, stop=5, step=0.1, value=2.0,
                           label="Logit (Cat)", show_value=True)
    ce_l1 = mo.ui.slider(start=-5, stop=5, step=0.1, value=1.0,
                           label="Logit (Dog)", show_value=True)
    ce_l2 = mo.ui.slider(start=-5, stop=5, step=0.1, value=0.5,
                           label="Logit (Bird)", show_value=True)
    mo.md(
        f"""
        ### Classification Cross-Entropy

        | | Control |
        |---|---------|
        | True class | {ce_true} |
        | Cat logit | {ce_l0} |
        | Dog logit | {ce_l1} |
        | Bird logit | {ce_l2} |
        """
    )
    return ce_l0, ce_l1, ce_l2, ce_true


@app.cell
def _(ce_l0, ce_l1, ce_l2, ce_true, np, plt, torch):
    _logits = torch.tensor([ce_l0.value, ce_l1.value, ce_l2.value])
    _target = torch.tensor(ce_true.value)
    _probs = torch.softmax(_logits, dim=0).numpy()
    _ce_loss = torch.nn.functional.cross_entropy(_logits.unsqueeze(0), _target.unsqueeze(0)).item()
    _labels = ['Cat', 'Dog', 'Bird']
    _true_idx = ce_true.value

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: softmax probabilities
    _colors = ['#3b82f6' if i != _true_idx else '#22c55e' for i in range(3)]
    _bars = _ax1.bar(_labels, _probs, color=_colors, edgecolor='white', linewidth=1.5)
    _ax1.set_ylabel('Probability')
    _ax1.set_title('Softmax Output q(x)', fontweight='bold')
    _ax1.set_ylim(0, 1)
    for _b, _p in zip(_bars, _probs):
        _ax1.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.02,
                 f'{_p:.3f}', ha='center', va='bottom', fontweight='bold')
    _ax1.axhline(1/3, color='gray', ls='--', alpha=0.3, label='uniform')
    _ax1.legend()

    # Right: loss landscape — sweep logit of true class
    _logit_range = np.linspace(-5, 5, 200)
    _losses = []
    for _l in _logit_range:
        _test_logits = [ce_l0.value, ce_l1.value, ce_l2.value]
        _test_logits[_true_idx] = _l
        _t = torch.tensor(_test_logits)
        _loss = torch.nn.functional.cross_entropy(_t.unsqueeze(0), _target.unsqueeze(0)).item()
        _losses.append(_loss)
    _ax2.plot(_logit_range, _losses, 'b-', lw=2)
    _current_logit = [ce_l0.value, ce_l1.value, ce_l2.value][_true_idx]
    _ax2.plot(_current_logit, _ce_loss, 'ro', markersize=10, zorder=5, label=f'Current: {_ce_loss:.4f}')
    _ax2.set_xlabel(f'Logit of true class ({_labels[_true_idx]})')
    _ax2.set_ylabel('Cross-Entropy Loss (nats)')
    _ax2.set_title('Loss vs True-Class Logit', fontweight='bold')
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    _fig.suptitle(f'Cross-Entropy Loss = {_ce_loss:.4f} nats', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 3: Cross-Entropy

    **Problem A (discrete):** True label = [1, 0, 0]. Model predicts q = [0.6, 0.3, 0.1]. Compute CE loss in nats.

    **Problem B (Gaussian):** p = N(0, 1), q = N(0, 4). Compute H(p, q) in nats.
    """)
    return


@app.cell
def _(mo):
    _q3a = mo.ui.number(start=0, stop=10, step=0.001, value=0.0,
                         label="Answer A: CE = ")
    _q3b = mo.ui.number(start=0, stop=10, step=0.001, value=0.0,
                         label="Answer B: H(p,q) = ")
    mo.md(f"""
    {_q3a} nats

    {_q3b} nats
    """)
    return


@app.cell
def _(mo):
    _show_q3 = mo.ui.switch(label="Show Solution", value=False)
    _show_q3
    return (_show_q3,)


@app.cell
def _(_show_q3, mo, np):
    if _show_q3.value:
        _ce_a = -np.log(0.6)
        _ce_b = 0.5 * np.log(2 * np.pi * 4) + (1 + 0) / (2 * 4)
        _H_p = 0.5 * np.log(2 * np.pi * np.e * 1)
        mo.md(
            f"""
            **Solution A:**
            For one-hot true label, CE = -log q(true class) = -ln(0.6) = **{_ce_a:.4f} nats**

            **Solution B:**
            H(p, q) = ½ ln(2π σ₂²) + (σ₁² + (μ₁-μ₂)²) / (2σ₂²)
            = ½ ln(2π × 4) + (1 + 0) / 8
            = ½ × {np.log(2*np.pi*4):.4f} + 0.125
            = {0.5*np.log(2*np.pi*4):.4f} + 0.125 = **{_ce_b:.4f} nats**

            Compare with H(p) = ½ ln(2πe) = {_H_p:.4f} nats.
            The gap H(p,q) - H(p) = {_ce_b - _H_p:.4f} nats is the KL divergence.
            """
        )
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 4: KL DIVERGENCE
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. KL Divergence

    $$D_{KL}(p \| q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)} = H(p, q) - H(p) \geq 0$$

    **Key property:** KL divergence is **not symmetric** — $D_{KL}(p \| q) \neq D_{KL}(q \| p)$.

    **Gaussian closed form:** $D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$
    """)
    return


@app.cell
def _(mo):
    kl_mu1 = mo.ui.slider(start=-3, stop=3, step=0.1, value=0.0,
                            label="μ₁ (p)", show_value=True)
    kl_s1 = mo.ui.slider(start=0.3, stop=4, step=0.1, value=1.0,
                           label="σ₁ (p)", show_value=True)
    kl_mu2 = mo.ui.slider(start=-3, stop=3, step=0.1, value=1.0,
                            label="μ₂ (q)", show_value=True)
    kl_s2 = mo.ui.slider(start=0.3, stop=4, step=0.1, value=2.0,
                           label="σ₂ (q)", show_value=True)
    mo.md(
        f"""
        ### Gaussian KL Divergence

        | | p (true) | q (model) |
        |---|---------|-----------|
        | Mean | {kl_mu1} | {kl_mu2} |
        | Std dev | {kl_s1} | {kl_s2} |
        """
    )
    return kl_mu1, kl_mu2, kl_s1, kl_s2


@app.cell
def _(D, kl_mu1, kl_mu2, kl_s1, kl_s2, np, plt, torch):
    _m1, _s1 = kl_mu1.value, kl_s1.value
    _m2, _s2 = kl_mu2.value, kl_s2.value

    # Closed-form KL
    _kl_fwd = np.log(_s2 / _s1) + (_s1**2 + (_m1 - _m2)**2) / (2 * _s2**2) - 0.5
    _kl_rev = np.log(_s1 / _s2) + (_s2**2 + (_m2 - _m1)**2) / (2 * _s1**2) - 0.5

    # Verify with PyTorch
    _p = D.Normal(torch.tensor(_m1), torch.tensor(_s1))
    _q = D.Normal(torch.tensor(_m2), torch.tensor(_s2))
    _kl_pt = D.kl_divergence(_p, _q).item()

    # Plot
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _x = np.linspace(min(_m1, _m2) - 4*max(_s1, _s2), max(_m1, _m2) + 4*max(_s1, _s2), 300)
    _p_pdf = (1/(_s1*np.sqrt(2*np.pi))) * np.exp(-0.5*((_x-_m1)/_s1)**2)
    _q_pdf = (1/(_s2*np.sqrt(2*np.pi))) * np.exp(-0.5*((_x-_m2)/_s2)**2)

    # Left: distributions
    _ax1.fill_between(_x, _p_pdf, alpha=0.3, color='blue', label=f'p ~ N({_m1}, {_s1**2:.1f})')
    _ax1.fill_between(_x, _q_pdf, alpha=0.3, color='orange', label=f'q ~ N({_m2}, {_s2**2:.1f})')
    _ax1.plot(_x, _p_pdf, 'b-', lw=2)
    _ax1.plot(_x, _q_pdf, 'orange', lw=2, ls='--')
    _ax1.set_title('Distributions', fontweight='bold')
    _ax1.legend()
    _ax1.set_ylabel('Density')
    _ax1.grid(True, alpha=0.3)

    # Right: pointwise KL contribution
    _kl_contrib = np.where(_p_pdf > 1e-10,
                           _p_pdf * np.log(np.maximum(_p_pdf, 1e-15) / np.maximum(_q_pdf, 1e-15)),
                           0)
    _ax2.fill_between(_x, _kl_contrib, alpha=0.5, color='purple', label='p(x) log(p/q)')
    _ax2.plot(_x, _kl_contrib, 'purple', lw=1.5)
    _ax2.axhline(0, color='gray', ls='-', alpha=0.3)
    _ax2.set_title('Pointwise KL Contribution', fontweight='bold')
    _ax2.set_xlabel('x')
    _ax2.set_ylabel('p(x) log(p(x)/q(x))')
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    _fig.suptitle(f'KL(p||q) = {_kl_fwd:.4f}  |  KL(q||p) = {_kl_rev:.4f}  |  Asymmetry = {abs(_kl_fwd-_kl_rev):.4f} nats',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 4: KL Divergence

    **Problem:** Compute $D_{KL}(p \| q)$ for discrete p = (0.5, 0.5) and q = (0.9, 0.1). Use log₂ (bits).
    Then compute $D_{KL}(q \| p)$. Are they equal?
    """)
    return


@app.cell
def _(mo):
    _q4a = mo.ui.number(start=0, stop=5, step=0.001, value=0.0,
                         label="D_KL(p||q) = ")
    _q4b = mo.ui.number(start=0, stop=5, step=0.001, value=0.0,
                         label="D_KL(q||p) = ")
    mo.md(f"""
    {_q4a} bits

    {_q4b} bits
    """)
    return


@app.cell
def _(mo):
    _show_q4 = mo.ui.switch(label="Show Solution", value=False)
    _show_q4
    return (_show_q4,)


@app.cell
def _(_show_q4, mo, np):
    if _show_q4.value:
        _kl_fwd = 0.5 * np.log2(0.5/0.9) + 0.5 * np.log2(0.5/0.1)
        _kl_rev = 0.9 * np.log2(0.9/0.5) + 0.1 * np.log2(0.1/0.5)
        mo.md(
            f"""
            **Solution:**

            D_KL(p||q) = 0.5 × log₂(0.5/0.9) + 0.5 × log₂(0.5/0.1)
            = 0.5 × ({np.log2(0.5/0.9):.4f}) + 0.5 × ({np.log2(0.5/0.1):.4f})
            = **{_kl_fwd:.4f} bits**

            D_KL(q||p) = 0.9 × log₂(0.9/0.5) + 0.1 × log₂(0.1/0.5)
            = 0.9 × ({np.log2(0.9/0.5):.4f}) + 0.1 × ({np.log2(0.1/0.5):.4f})
            = **{_kl_rev:.4f} bits**

            **Not equal!** {_kl_fwd:.4f} ≠ {_kl_rev:.4f} — KL divergence is asymmetric.
            """
        )
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 5: MLE
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Maximum Likelihood Estimation (MLE)

    Find parameters that maximize $\mathcal{L}(\theta) = \prod_i p(x_i \mid \theta)$

    Equivalently, maximize **log-likelihood**: $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$

    | Distribution | MLE |
    |---|---|
    | Gaussian mean | $\hat{\mu} = \bar{x}$ (sample mean) |
    | Gaussian variance | $\hat{\sigma}^2 = \frac{1}{N}\sum(x_i - \bar{x})^2$ |
    | Bernoulli | $\hat{\theta} = \frac{\text{successes}}{N}$ |
    """)
    return


@app.cell
def _(mo):
    mle_mu_true = mo.ui.slider(start=-3, stop=3, step=0.1, value=2.0,
                                 label="True μ", show_value=True)
    mle_sigma_true = mo.ui.slider(start=0.5, stop=3, step=0.1, value=1.0,
                                    label="True σ", show_value=True)
    mle_n = mo.ui.slider(start=5, stop=500, step=5, value=50,
                           label="N (sample size)", show_value=True)
    mle_seed = mo.ui.slider(start=1, stop=100, step=1, value=42,
                              label="Random seed", show_value=True)
    mo.md(
        f"""
        ### MLE for Gaussian — Live Demo

        | Parameter | Control |
        |-----------|---------|
        | True μ | {mle_mu_true} |
        | True σ | {mle_sigma_true} |
        | Sample size | {mle_n} |
        | Seed | {mle_seed} |
        """
    )
    return mle_mu_true, mle_n, mle_seed, mle_sigma_true


@app.cell
def _(mle_mu_true, mle_n, mle_seed, mle_sigma_true, np, plt):
    _mu_t = mle_mu_true.value
    _sig_t = mle_sigma_true.value
    _n = int(mle_n.value)
    _rng = np.random.default_rng(int(mle_seed.value))
    _data = _rng.normal(_mu_t, _sig_t, _n)

    # MLE estimates
    _mu_mle = np.mean(_data)
    _var_mle = np.mean((_data - _mu_mle)**2)  # MLE uses N, not N-1
    _sig_mle = np.sqrt(_var_mle)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: data + fitted Gaussian
    _x = np.linspace(_mu_t - 4*_sig_t, _mu_t + 4*_sig_t, 300)
    _true_pdf = (1/(_sig_t*np.sqrt(2*np.pi))) * np.exp(-0.5*((_x-_mu_t)/_sig_t)**2)
    _mle_pdf = (1/(_sig_mle*np.sqrt(2*np.pi))) * np.exp(-0.5*((_x-_mu_mle)/_sig_mle)**2)

    _ax1.hist(_data, bins=min(30, _n//3), density=True, alpha=0.5, color='steelblue', label='Data')
    _ax1.plot(_x, _true_pdf, 'g-', lw=2, label=f'True N({_mu_t}, {_sig_t**2:.1f})')
    _ax1.plot(_x, _mle_pdf, 'r--', lw=2, label=f'MLE N({_mu_mle:.2f}, {_var_mle:.2f})')
    _ax1.set_title(f'MLE Fit (N={_n})', fontweight='bold')
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    # Right: log-likelihood surface over mu
    _mus = np.linspace(_mu_t - 3, _mu_t + 3, 200)
    _lls = np.array([-_n/2 * np.log(2*np.pi*_sig_t**2) - np.sum((_data - m)**2)/(2*_sig_t**2)
                      for m in _mus])
    _ax2.plot(_mus, _lls, 'b-', lw=2)
    _ax2.axvline(_mu_mle, color='red', ls='--', lw=1.5, label=f'μ_MLE = {_mu_mle:.3f}')
    _ax2.axvline(_mu_t, color='green', ls=':', lw=1.5, label=f'μ_true = {_mu_t:.1f}')
    _ax2.set_xlabel('μ')
    _ax2.set_ylabel('Log-likelihood ℓ(μ)')
    _ax2.set_title('Log-Likelihood Landscape', fontweight='bold')
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    _fig.suptitle(f'MLE: μ̂={_mu_mle:.3f} (true {_mu_t}), σ̂={_sig_mle:.3f} (true {_sig_t})',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 5: MLE

    **Problem A:** You flip a coin 80 times and get 60 heads. What is $\hat{\theta}_{MLE}$?

    **Problem B:** Given samples {3, 5, 7, 5, 10}, what is $\hat{\mu}_{MLE}$ and $\hat{\sigma}^2_{MLE}$?
    """)
    return


@app.cell
def _(mo):
    _q5a = mo.ui.number(start=0, stop=1, step=0.001, value=0.0,
                         label="Answer A: θ_MLE = ")
    _q5b_mu = mo.ui.number(start=-10, stop=20, step=0.01, value=0.0,
                            label="Answer B: μ_MLE = ")
    _q5b_var = mo.ui.number(start=0, stop=50, step=0.01, value=0.0,
                             label="Answer B: σ²_MLE = ")
    mo.md(f"""
    {_q5a}

    {_q5b_mu}

    {_q5b_var}
    """)
    return


@app.cell
def _(mo):
    _show_q5 = mo.ui.switch(label="Show Solution", value=False)
    _show_q5
    return (_show_q5,)


@app.cell
def _(_show_q5, mo, np):
    if _show_q5.value:
        _data = np.array([3, 5, 7, 5, 10])
        _mu = np.mean(_data)
        _var = np.mean((_data - _mu)**2)
        mo.md(
            f"""
            **Solution A:**
            $\\hat{{\\theta}}_{{MLE}}$ = successes / N = 60 / 80 = **0.750**

            **Solution B:**
            $\\hat{{\\mu}}_{{MLE}}$ = (3+5+7+5+10) / 5 = 30/5 = **{_mu:.1f}**

            $\\hat{{\\sigma}}^2_{{MLE}}$ = [(3-6)² + (5-6)² + (7-6)² + (5-6)² + (10-6)²] / 5
            = [9 + 1 + 1 + 1 + 16] / 5 = 28/5 = **{_var:.1f}**

            Note: MLE divides by N={len(_data)}, giving {_var:.1f}.
            The unbiased estimator divides by N-1={len(_data)-1}, giving {np.var(_data, ddof=1):.2f}.
            """
        )
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 6: MAP
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. MAP Estimation

    $$\hat{\theta}_{MAP} = \arg\max_\theta \left[\sum_i \log p(x_i \mid \theta) + \log p(\theta)\right]$$

    MAP = MLE + prior. For Bernoulli with Beta(α, β) prior:

    $$\hat{\theta}_{MAP} = \frac{k + \alpha - 1}{N + \alpha + \beta - 2}$$

    Below, see how prior strength pulls the estimate toward the prior mean.
    """)
    return


@app.cell
def _(mo):
    map_k = mo.ui.slider(start=0, stop=20, step=1, value=3,
                           label="k (heads observed)", show_value=True)
    map_n = mo.ui.slider(start=1, stop=20, step=1, value=3,
                           label="N (total flips)", show_value=True)
    map_alpha = mo.ui.slider(start=1, stop=20, step=0.5, value=3.0,
                               label="α (Beta prior)", show_value=True)
    map_beta = mo.ui.slider(start=1, stop=20, step=0.5, value=3.0,
                              label="β (Beta prior)", show_value=True)
    mo.md(
        f"""
        ### Bernoulli MLE vs MAP (Beta Prior)

        | Parameter | Control |
        |-----------|---------|
        | Heads (k) | {map_k} |
        | Total flips (N) | {map_n} |
        | Prior α | {map_alpha} |
        | Prior β | {map_beta} |
        """
    )
    return map_alpha, map_beta, map_k, map_n


@app.cell
def _(map_alpha, map_beta, map_k, map_n, np, plt, stats):
    _k = min(map_k.value, map_n.value)  # can't have more heads than flips
    _n = map_n.value
    _a = map_alpha.value
    _b = map_beta.value

    # Estimates
    _mle = _k / _n if _n > 0 else 0.5
    _map = (_k + _a - 1) / (_n + _a + _b - 2) if (_n + _a + _b - 2) > 0 else 0.5
    _map = np.clip(_map, 0.001, 0.999)
    _prior_mean = _a / (_a + _b)

    # Posterior: Beta(k + α, N - k + β)
    _post_a = _k + _a
    _post_b = (_n - _k) + _b

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _theta = np.linspace(0.001, 0.999, 300)

    # Left: Prior, Likelihood, Posterior
    _prior_pdf = stats.beta.pdf(_theta, _a, _b)
    _like = _theta**_k * (1 - _theta)**(_n - _k)
    _like = _like / (_like.max() + 1e-15) * _prior_pdf.max()  # scale for visibility
    _post_pdf = stats.beta.pdf(_theta, _post_a, _post_b)

    _ax1.plot(_theta, _prior_pdf, 'b-', lw=2, label=f'Prior Beta({_a:.0f},{_b:.0f})')
    _ax1.plot(_theta, _like, 'g--', lw=2, label=f'Likelihood (scaled)')
    _ax1.fill_between(_theta, _post_pdf, alpha=0.3, color='red')
    _ax1.plot(_theta, _post_pdf, 'r-', lw=2, label=f'Posterior Beta({_post_a:.0f},{_post_b:.0f})')
    _ax1.axvline(_mle, color='green', ls=':', lw=2, label=f'MLE = {_mle:.3f}')
    _ax1.axvline(_map, color='red', ls=':', lw=2, label=f'MAP = {_map:.3f}')
    _ax1.set_xlabel('θ')
    _ax1.set_ylabel('Density')
    _ax1.set_title('Prior × Likelihood → Posterior', fontweight='bold')
    _ax1.legend(fontsize=8)
    _ax1.set_xlim(0, 1)
    _ax1.grid(True, alpha=0.3)

    # Right: MLE vs MAP as N increases
    _Ns = np.arange(1, 101)
    _mles = []
    _maps = []
    for _ni in _Ns:
        _ki = int(round(_mle * _ni))  # keep same proportion
        _mles.append(_ki / _ni)
        _denom = _ni + _a + _b - 2
        _maps.append((_ki + _a - 1) / _denom if _denom > 0 else 0.5)

    _ax2.plot(_Ns, _mles, 'g-', lw=2, label='MLE')
    _ax2.plot(_Ns, _maps, 'r-', lw=2, label='MAP')
    _ax2.axhline(_prior_mean, color='blue', ls='--', alpha=0.5, label=f'Prior mean = {_prior_mean:.2f}')
    _ax2.axhline(_mle, color='gray', ls=':', alpha=0.5, label=f'Data proportion = {_mle:.2f}')
    _ax2.set_xlabel('N (sample size)')
    _ax2.set_ylabel('θ estimate')
    _ax2.set_title('MLE vs MAP Convergence', fontweight='bold')
    _ax2.legend(fontsize=9)
    _ax2.grid(True, alpha=0.3)
    _ax2.set_ylim(0, 1)

    _fig.suptitle(f'Data: {_k}/{_n} heads | MLE={_mle:.3f} | MAP={_map:.3f} | Prior mean={_prior_mean:.2f}',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Quiz 6: MAP

    **Problem:** You flip a coin 5 times, get 5 heads (k=5, N=5). Your prior is Beta(2, 2).

    a) What is θ_MLE?
    b) What is θ_MAP?
    c) What is the prior mean?
    d) Which estimate is more reasonable and why?
    """)
    return


@app.cell
def _(mo):
    _q6a = mo.ui.number(start=0, stop=1, step=0.001, value=0.0, label="θ_MLE = ")
    _q6b = mo.ui.number(start=0, stop=1, step=0.001, value=0.0, label="θ_MAP = ")
    _q6c = mo.ui.number(start=0, stop=1, step=0.001, value=0.0, label="Prior mean = ")
    mo.md(f"""
    {_q6a}

    {_q6b}

    {_q6c}
    """)
    return


@app.cell
def _(mo):
    _show_q6 = mo.ui.switch(label="Show Solution", value=False)
    _show_q6
    return (_show_q6,)


@app.cell
def _(_show_q6, mo):
    if _show_q6.value:
        mo.md(
            r"""
            **Solution:**

            a) θ_MLE = k/N = 5/5 = **1.000** (claims coin always lands heads)

            b) θ_MAP = (k + α - 1)/(N + α + β - 2) = (5 + 2 - 1)/(5 + 2 + 2 - 2) = 6/7 = **0.857**

            c) Prior mean = α/(α + β) = 2/4 = **0.500**

            d) **MAP is more reasonable.** MLE = 1.0 claims the coin is perfectly biased
            (never lands tails), which is extreme given only 5 observations.
            The Beta(2,2) prior encodes a mild belief that coins tend to be near fair,
            pulling the estimate to 0.857 — still biased toward heads (as the data suggests)
            but not maximally so. With more data, both estimates would converge.
            """
        )
    else:
        mo.md("*Toggle the switch above to reveal the solution.*")
    return


# ═══════════════════════════════════════════════════════════════
# SECTION 7: SUMMARY
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7. Chapter 6 Summary — The Big Picture

    ```
    True distribution p(x)
            │
            ▼
        Sample data D = {x₁, ..., xₙ}
            │
            ▼
        Model q_θ(x) with parameters θ
            │
            ▼
        Minimize cross-entropy H(p, q_θ)
            ≡ Maximize log-likelihood ℓ(θ)         ← MLE
            ≡ Minimize KL divergence D_KL(p ∥ q_θ)
            │
        + log P(θ) prior
            │
            ▼
        MAP estimation ← equivalent to regularized loss
    ```

    ### Key Equivalences

    | What you call it | What it really is |
    |---|---|
    | Cross-entropy loss | Negative log-likelihood (MLE) |
    | MSE loss | Negative log-likelihood under Gaussian noise |
    | L2 regularization (weight decay) | MAP with Gaussian prior on weights |
    | L1 regularization (lasso) | MAP with Laplace prior on weights |
    | KL term in VAE loss | Regularization toward N(0, I) prior |

    ### Connections Across Chapters

    | Chapter 5 (Distributions) | Chapter 6 (Bayesian Tools) |
    |---|---|
    | Gaussian N(μ, σ²) | Entropy = ½ ln(2πeσ²); MLE for μ and σ² |
    | Bernoulli Ber(θ) | MLE: θ̂ = k/N; MAP with Beta prior |
    | Categorical Cat(θ) | Cross-entropy loss for multi-class |
    | Covariance matrix Σ | KL divergence between multivariate Gaussians |

    ---

    **Next up (Chapter 7):** These tools power neural network training — cross-entropy loss,
    softmax output, backpropagation, and regularization all trace back to MLE/MAP.
    """)
    return


if __name__ == "__main__":
    app.run()
