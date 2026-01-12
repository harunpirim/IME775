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
    # Week 13: Attention Mechanisms and Transformers

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 11

    ---

    ## Learning Objectives

    - Understand the attention mechanism mathematically
    - Master self-attention and multi-head attention
    - Learn the complete Transformer architecture
    - Connect to modern applications (BERT, GPT)
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.1 The Attention Mechanism

    **Core Idea**: Given a query, compute relevance scores over keys to retrieve values.

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
    """)
    return


@app.cell
def _(np, plt):
    # Implement and visualize attention
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    def attention(Q, K, V):
        """Scaled dot-product attention."""
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores, axis=-1)
        output = weights @ V
        return output, weights

    # Example: Simple sentence attention
    # "The cat sat on the mat"
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(words)
    d_model = 8

    # Create random embeddings (in practice these would be learned)
    np.random.seed(42)
    embeddings = np.random.randn(n_words, d_model)

    # Self-attention: Q, K, V all from same sequence
    Q = K = V = embeddings

    output, attn_weights = attention(Q, K, V)

    # Visualize attention weights
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Attention matrix
    im1 = axes1[0].imshow(attn_weights, cmap='Blues')
    axes1[0].set_xticks(range(n_words))
    axes1[0].set_yticks(range(n_words))
    axes1[0].set_xticklabels(words, rotation=45, ha='right')
    axes1[0].set_yticklabels(words)
    axes1[0].set_xlabel('Key (attending to)')
    axes1[0].set_ylabel('Query (from)')
    axes1[0].set_title('Self-Attention Weights')
    plt.colorbar(im1, ax=axes1[0])

    # Show attention for specific word
    query_idx = 2  # "sat"
    axes1[1].bar(range(n_words), attn_weights[query_idx], color='steelblue')
    axes1[1].set_xticks(range(n_words))
    axes1[1].set_xticklabels(words)
    axes1[1].set_ylabel('Attention Weight')
    axes1[1].set_title(f'Attention from "{words[query_idx]}" to all words')
    axes1[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig1
    return (
        K,
        Q,
        V,
        attention,
        attn_weights,
        axes1,
        d_model,
        embeddings,
        fig1,
        im1,
        n_words,
        output,
        query_idx,
        softmax,
        words,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.2 Multi-Head Attention

    Multiple attention heads can capture different relationship types:

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

    Each head: $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
    """)
    return


@app.cell
def _(attention, d_model, embeddings, n_words, np, plt, softmax, words):
    # Multi-head attention visualization
    def multi_head_attention(X, n_heads=4):
        """Simplified multi-head attention."""
        d_k = d_model // n_heads
        
        all_weights = []
        
        for h in range(n_heads):
            # Random projections (in practice, learned)
            np.random.seed(42 + h)
            W_q = np.random.randn(d_model, d_k) * 0.1
            W_k = np.random.randn(d_model, d_k) * 0.1
            W_v = np.random.randn(d_model, d_k) * 0.1
            
            Q_h = X @ W_q
            K_h = X @ W_k
            V_h = X @ W_v
            
            _, weights = attention(Q_h, K_h, V_h)
            all_weights.append(weights)
        
        return all_weights

    n_heads = 4
    head_weights = multi_head_attention(embeddings, n_heads)

    fig2, axes2 = plt.subplots(1, n_heads, figsize=(16, 4))

    for h, weights in enumerate(head_weights):
        im = axes2[h].imshow(weights, cmap='Blues')
        axes2[h].set_xticks(range(n_words))
        axes2[h].set_yticks(range(n_words))
        axes2[h].set_xticklabels(words, rotation=45, ha='right', fontsize=8)
        axes2[h].set_yticklabels(words, fontsize=8)
        axes2[h].set_title(f'Head {h+1}', fontsize=12)

    plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns', fontsize=14, y=1.05)
    plt.tight_layout()
    fig2
    return (
        axes2,
        fig2,
        h,
        head_weights,
        im,
        multi_head_attention,
        n_heads,
        weights,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.3 Positional Encoding

    Self-attention is permutation-invariant - it doesn't know position!

    **Solution**: Add positional information:
    $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
    $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$
    """)
    return


@app.cell
def _(np, plt):
    # Visualize positional encoding
    def positional_encoding(max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe

    max_len = 100
    d_pe = 64
    pe = positional_encoding(max_len, d_pe)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Full encoding
    im3 = axes3[0].imshow(pe.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes3[0].set_xlabel('Position')
    axes3[0].set_ylabel('Dimension')
    axes3[0].set_title('Positional Encoding Matrix')
    plt.colorbar(im3, ax=axes3[0])

    # Individual dimensions
    for dim in [0, 1, 10, 20]:
        axes3[1].plot(pe[:, dim], label=f'dim {dim}', linewidth=1.5)
    axes3[1].set_xlabel('Position')
    axes3[1].set_ylabel('Encoding Value')
    axes3[1].set_title('Positional Encoding: Different Dimensions')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3
    return axes3, d_pe, dim, fig3, im3, max_len, pe, positional_encoding


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.4 The Transformer Architecture

    ```
    Input → Embedding + Position → [Encoder × N] → [Decoder × N] → Output
    
    Encoder:                    Decoder:
    ┌─────────────────┐        ┌─────────────────┐
    │ Multi-Head Attn │        │ Masked Self-Attn│
    │   + Add & Norm  │        │   + Add & Norm  │
    ├─────────────────┤        ├─────────────────┤
    │                 │        │ Cross-Attention │
    │                 │        │   + Add & Norm  │
    ├─────────────────┤        ├─────────────────┤
    │ Feed-Forward    │        │ Feed-Forward    │
    │   + Add & Norm  │        │   + Add & Norm  │
    └─────────────────┘        └─────────────────┘
    ```
    """)
    return


@app.cell
def _(np, plt):
    # Visualize Transformer architecture
    fig4, ax4 = plt.subplots(figsize=(14, 10))

    # Encoder stack
    enc_x = 3
    for i in range(3):
        y_base = 2 + i * 2.5
        
        # Multi-head attention
        ax4.add_patch(plt.Rectangle((enc_x - 1, y_base), 2, 0.8, 
                                     facecolor='lightblue', edgecolor='black'))
        ax4.text(enc_x, y_base + 0.4, 'Multi-Head\nSelf-Attention', 
                ha='center', va='center', fontsize=8)
        
        # Add & Norm
        ax4.add_patch(plt.Rectangle((enc_x - 1, y_base + 0.9), 2, 0.3,
                                     facecolor='lightgray', edgecolor='black'))
        ax4.text(enc_x, y_base + 1.05, 'Add & Norm', ha='center', va='center', fontsize=7)
        
        # FFN
        ax4.add_patch(plt.Rectangle((enc_x - 1, y_base + 1.3), 2, 0.6,
                                     facecolor='lightyellow', edgecolor='black'))
        ax4.text(enc_x, y_base + 1.6, 'Feed Forward', ha='center', va='center', fontsize=8)
        
        # Add & Norm
        ax4.add_patch(plt.Rectangle((enc_x - 1, y_base + 2.0), 2, 0.3,
                                     facecolor='lightgray', edgecolor='black'))
        ax4.text(enc_x, y_base + 2.15, 'Add & Norm', ha='center', va='center', fontsize=7)
        
        # Residual connections
        ax4.plot([enc_x - 1.3, enc_x - 1.3], [y_base + 0.4, y_base + 1.05], 'r-', lw=1)
        ax4.plot([enc_x - 1.3, enc_x - 1], [y_base + 1.05, y_base + 1.05], 'r-', lw=1)

    # Decoder stack
    dec_x = 9
    for i in range(3):
        y_base = 2 + i * 2.5
        
        # Masked self-attention
        ax4.add_patch(plt.Rectangle((dec_x - 1, y_base), 2, 0.6,
                                     facecolor='lightgreen', edgecolor='black'))
        ax4.text(dec_x, y_base + 0.3, 'Masked\nSelf-Attn', ha='center', va='center', fontsize=7)
        
        # Cross-attention
        ax4.add_patch(plt.Rectangle((dec_x - 1, y_base + 0.7), 2, 0.6,
                                     facecolor='lightcoral', edgecolor='black'))
        ax4.text(dec_x, y_base + 1.0, 'Cross\nAttention', ha='center', va='center', fontsize=7)
        
        # FFN
        ax4.add_patch(plt.Rectangle((dec_x - 1, y_base + 1.4), 2, 0.5,
                                     facecolor='lightyellow', edgecolor='black'))
        ax4.text(dec_x, y_base + 1.65, 'FFN', ha='center', va='center', fontsize=8)
        
        # Arrow from encoder to cross-attention
        if i == 2:
            ax4.annotate('', xy=(dec_x - 1, y_base + 1.0), xytext=(enc_x + 1, y_base + 1.0),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=2))

    # Input
    ax4.add_patch(plt.Rectangle((enc_x - 1, 0.5), 2, 0.8, facecolor='white', edgecolor='black'))
    ax4.text(enc_x, 0.9, 'Input\nEmbedding\n+ Position', ha='center', va='center', fontsize=8)

    ax4.add_patch(plt.Rectangle((dec_x - 1, 0.5), 2, 0.8, facecolor='white', edgecolor='black'))
    ax4.text(dec_x, 0.9, 'Output\nEmbedding\n+ Position', ha='center', va='center', fontsize=8)

    # Output
    ax4.add_patch(plt.Rectangle((dec_x - 1, 10), 2, 0.8, facecolor='white', edgecolor='black'))
    ax4.text(dec_x, 10.4, 'Linear +\nSoftmax', ha='center', va='center', fontsize=8)

    # Labels
    ax4.text(enc_x, 11, 'ENCODER', ha='center', fontsize=14, fontweight='bold', color='blue')
    ax4.text(dec_x, 11, 'DECODER', ha='center', fontsize=14, fontweight='bold', color='green')

    # Arrows
    ax4.annotate('', xy=(enc_x, 2), xytext=(enc_x, 1.3), arrowprops=dict(arrowstyle='->', color='black'))
    ax4.annotate('', xy=(dec_x, 2), xytext=(dec_x, 1.3), arrowprops=dict(arrowstyle='->', color='black'))
    ax4.annotate('', xy=(dec_x, 10), xytext=(dec_x, 9.5), arrowprops=dict(arrowstyle='->', color='black'))

    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 12)
    ax4.axis('off')
    ax4.set_title('Transformer Architecture (Simplified)', fontsize=14, y=0.98)

    fig4
    return ax4, dec_x, enc_x, fig4, i, y_base


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.5 Masked Attention (Decoder)

    Prevent attending to future positions during training:

    $$\text{Mask}_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$
    """)
    return


@app.cell
def _(np, plt, softmax):
    # Demonstrate causal masking
    def masked_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
        
        weights = softmax(scores, axis=-1)
        output = weights @ V
        return output, weights

    # Create causal mask
    seq_len = 6
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)

    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4))

    # Causal mask
    mask_display = np.where(causal_mask < -1e8, -np.inf, 0)
    axes5[0].imshow(mask_display, cmap='RdYlGn', vmin=-1, vmax=0)
    for i in range(seq_len):
        for j in range(seq_len):
            val = '0' if causal_mask[i, j] > -1e8 else '-∞'
            color = 'black' if val == '0' else 'red'
            axes5[0].text(j, i, val, ha='center', va='center', fontsize=12, color=color)
    axes5[0].set_title('Causal Mask\n(Green=Allowed, Red=Blocked)')
    axes5[0].set_xlabel('Key position')
    axes5[0].set_ylabel('Query position')

    # Without mask
    np.random.seed(42)
    embeddings_mask = np.random.randn(seq_len, 8)
    _, weights_no_mask = masked_attention(embeddings_mask, embeddings_mask, embeddings_mask, mask=None)
    axes5[1].imshow(weights_no_mask, cmap='Blues')
    axes5[1].set_title('Attention WITHOUT Mask\n(Can see future)')
    axes5[1].set_xlabel('Key position')
    axes5[1].set_ylabel('Query position')

    # With mask
    _, weights_with_mask = masked_attention(embeddings_mask, embeddings_mask, embeddings_mask, mask=causal_mask)
    axes5[2].imshow(weights_with_mask, cmap='Blues')
    axes5[2].set_title('Attention WITH Causal Mask\n(Can only see past)')
    axes5[2].set_xlabel('Key position')
    axes5[2].set_ylabel('Query position')

    plt.tight_layout()
    fig5
    return (
        axes5,
        causal_mask,
        color,
        embeddings_mask,
        fig5,
        i,
        j,
        mask_display,
        masked_attention,
        seq_len,
        val,
        weights_no_mask,
        weights_with_mask,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 13.6 Modern Transformer Variants

    | Model | Type | Pre-training Task | Use Case |
    |-------|------|-------------------|----------|
    | **BERT** | Encoder | Masked LM + Next Sentence | Classification, NER |
    | **GPT** | Decoder | Next Token Prediction | Generation, Chat |
    | **T5** | Encoder-Decoder | Text-to-Text | Translation, Summarization |
    | **ViT** | Encoder | Image Classification | Vision tasks |
    """)
    return


@app.cell
def _(np, plt):
    # Compare architectures
    fig6, ax6 = plt.subplots(figsize=(12, 6))

    models = ['BERT-base', 'BERT-large', 'GPT-2', 'GPT-3', 'T5-base', 'ViT-base']
    params = [110, 340, 1500, 175000, 220, 86]  # In millions
    colors = ['blue', 'blue', 'green', 'green', 'orange', 'purple']

    bars = ax6.bar(models, params, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Parameters (Millions)')
    ax6.set_title('Transformer Model Sizes')
    ax6.set_yscale('log')

    # Add parameter counts
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Legend
    ax6.text(0.02, 0.95, '■ BERT (Encoder)', color='blue', transform=ax6.transAxes, fontsize=10)
    ax6.text(0.02, 0.90, '■ GPT (Decoder)', color='green', transform=ax6.transAxes, fontsize=10)
    ax6.text(0.02, 0.85, '■ T5 (Enc-Dec)', color='orange', transform=ax6.transAxes, fontsize=10)
    ax6.text(0.02, 0.80, '■ ViT (Vision)', color='purple', transform=ax6.transAxes, fontsize=10)

    ax6.grid(True, alpha=0.3, axis='y')

    fig6
    return ax6, bar, bars, colors, fig6, height, models, param, params


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Component | Purpose |
    |-----------|---------|
    | **Attention** | Compute relevance-weighted combinations |
    | **Self-Attention** | Relate positions within sequence |
    | **Multi-Head** | Capture multiple relationship types |
    | **Positional Encoding** | Add position information |
    | **Masking** | Prevent attending to future |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 11.
    - **Transformer**: Vaswani et al. (2017). "Attention Is All You Need."
    - **BERT**: Devlin et al. (2019)
    - **GPT-3**: Brown et al. (2020)

    ## Connection to ML Refined Curriculum

    Transformers are the current state-of-the-art for:
    - Sequence modeling (beyond RNNs from Week 11)
    - Automatic feature learning
    - Foundation for modern AI systems
    """)
    return


if __name__ == "__main__":
    app.run()

