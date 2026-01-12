# Week 13: Attention Mechanisms and Transformers

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 11: Attention and Transformers**.

---

## Overview

This module covers attention mechanisms and the Transformer architecture, the foundation of modern NLP and increasingly computer vision.

---

## Learning Objectives

- Understand the attention mechanism mathematically
- Master self-attention and multi-head attention
- Learn the complete Transformer architecture
- Connect transformers to modern applications (BERT, GPT)

---

## 13.1 Motivation: Beyond RNNs

### RNN Limitations

| Problem | Description |
|---------|-------------|
| Sequential processing | Can't parallelize over time |
| Long-range dependencies | Even LSTM struggles with very long sequences |
| Fixed context | Hard to attend to specific relevant parts |

### The Attention Solution

Allow the model to **look at all positions** and **weight them by relevance**.

---

## 13.2 Attention Mechanism

### Core Idea

Given a **query**, compute relevance scores over a set of **keys** to retrieve **values**.

### Attention Function

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix
- $\sqrt{d_k}$: Scaling factor

### Step-by-Step

1. **Compute scores**: $S = QK^T$ (how relevant is each key to each query)
2. **Scale**: $S = S / \sqrt{d_k}$ (prevent large values)
3. **Softmax**: $A = \text{softmax}(S)$ (normalize to probabilities)
4. **Weighted sum**: $\text{output} = AV$ (retrieve values by weights)

### Why Scaling?

Without $\sqrt{d_k}$, dot products can be large → softmax saturates → gradients vanish.

---

## 13.3 Self-Attention

### Definition

Attention where queries, keys, and values all come from the **same sequence**.

For input sequence $X \in \mathbb{R}^{n \times d}$:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

Where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ are learned projections.

### What It Computes

Each position attends to all other positions:
- Position $i$ looks at positions $1, 2, \ldots, n$
- Weights depend on content (not just position)

### Example: "The cat sat on the mat"

For word "sat":
- High attention to "cat" (subject)
- High attention to "mat" (related via preposition)
- Lower attention to articles

---

## 13.4 Multi-Head Attention

### Motivation

Single attention head has limited expressiveness. Multiple heads can attend to different aspects.

### Mechanism

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Intuition

Different heads can learn to attend to:
- Syntactic relationships
- Semantic relationships
- Positional patterns
- etc.

### Typical Configuration

- 8 or 12 heads
- $d_k = d_{model} / h$ (e.g., 512/8 = 64)

---

## 13.5 Positional Encoding

### Problem

Self-attention is permutation-invariant - it doesn't know position!

### Solution: Add Position Information

$$\text{input} = \text{embedding} + \text{positional encoding}$$

### Sinusoidal Encoding (Original Transformer)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

### Properties

- Unique encoding for each position
- Can extrapolate to longer sequences
- Relative positions can be computed from encodings

### Learned Positional Embeddings

Alternative: Learn position embeddings (like word embeddings).

---

## 13.6 The Transformer Architecture

### Encoder

```
Input Embedding + Positional Encoding
         ↓
    [Multi-Head Self-Attention]
         ↓  (+ residual)
    [Layer Norm]
         ↓
    [Feed-Forward Network]
         ↓  (+ residual)
    [Layer Norm]
         ↓
    (Repeat N times)
```

### Decoder

Same as encoder, plus:
1. **Masked self-attention**: Can only attend to previous positions
2. **Cross-attention**: Attend to encoder output

```
Output Embedding + Positional Encoding
         ↓
    [Masked Multi-Head Self-Attention]
         ↓  (+ residual + norm)
    [Multi-Head Cross-Attention (to encoder)]
         ↓  (+ residual + norm)
    [Feed-Forward Network]
         ↓  (+ residual + norm)
    (Repeat N times)
```

### Feed-Forward Network

Position-wise (same for each position):
$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Typically: $d_{ff} = 4 \times d_{model}$

### Why Layer Normalization?

- Stabilizes training
- Applied after each sub-layer (with residual)

---

## 13.7 Masked Attention (Decoder)

### Purpose

During training, prevent attending to future positions (no cheating!).

### Implementation

Set attention scores to $-\infty$ for future positions before softmax:

$$\text{Mask}_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

---

## 13.8 Training and Inference

### Training

- Teacher forcing: Use ground truth as decoder input
- Cross-entropy loss on output tokens
- Train all positions in parallel

### Inference (Autoregressive)

```
1. Encode input sequence
2. Start with <SOS> token
3. For each position:
   - Run decoder
   - Sample/argmax next token
   - Append to decoder input
4. Stop at <EOS> or max length
```

---

## 13.9 Modern Variants

### Encoder-Only: BERT

- Pre-trained on masked language modeling
- Fine-tuned for classification, NER, etc.
- Bidirectional (sees all context)

### Decoder-Only: GPT

- Pre-trained on next-token prediction
- Autoregressive generation
- Unidirectional (causal attention)

### Encoder-Decoder: T5, BART

- Sequence-to-sequence tasks
- Translation, summarization

### Vision Transformer (ViT)

Apply transformers to images:
- Split image into patches
- Treat patches as "tokens"
- Add position embeddings
- Standard transformer encoder

---

## 13.10 Implementation

### Scaled Dot-Product Attention

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """
    Q: (batch, n_queries, d_k)
    K: (batch, n_keys, d_k)
    V: (batch, n_keys, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = scores + mask  # mask has -inf for blocked positions
    
    # Softmax
    weights = softmax(scores, axis=-1)
    
    # Weighted sum
    output = weights @ V
    
    return output, weights
```

### Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Project
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention
        attn_output, _ = attention(
            Q.reshape(-1, Q.shape[2], Q.shape[3]),
            K.reshape(-1, K.shape[2], K.shape[3]),
            V.reshape(-1, V.shape[2], V.shape[3]),
            mask
        )
        
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, self.n_heads, -1, self.d_k)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.d_k)
        
        return attn_output @ self.W_o
```

### Positional Encoding

```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
```

---

## Exercises

### Exercise 13.1 (Section 13.2)
Implement scaled dot-product attention from scratch and visualize attention weights.

### Exercise 13.2 (Section 13.5)
Plot positional encodings and show that relative positions can be computed.

### Exercise 13.3 (Section 13.6)
Build a simple transformer encoder layer (no decoder).

### Exercise 13.4 (Section 13.7)
Implement causal masking and verify future positions can't be attended to.

---

## Summary

- Attention allows models to focus on relevant parts of input
- Self-attention relates positions within a sequence
- Multi-head attention captures different relationship types
- Transformers use attention for both encoding and decoding
- Modern variants: BERT (encoder), GPT (decoder), T5 (encoder-decoder)

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 11.

### Key Papers
- Vaswani, A., et al. (2017). "Attention Is All You Need."
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." (GPT-3)

---

## Course Progression

Transformers are the culmination of the course:
- **Builds on**: Linear algebra (Week 1), optimization (Week 3), backprop (Week 5)
- **Extends**: Sequence modeling beyond RNNs (Week 11)
- **Applications**: NLP (BERT, GPT), Vision (ViT), multimodal models
- **Projects**: Foundation for working with modern AI systems

