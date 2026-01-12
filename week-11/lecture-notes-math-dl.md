# Week 11: Recurrent Neural Networks (RNNs) and Sequence Models

## Reference

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications. **Chapter 10: Recurrent Neural Networks**.

---

## Overview

This module covers recurrent neural networks for sequential data, including vanilla RNNs, LSTMs, and GRUs.

---

## Learning Objectives

- Understand sequence modeling challenges
- Master RNN architecture and mathematics
- Learn LSTM and GRU mechanisms for long-term dependencies
- Implement sequence-to-sequence models

---

## 11.1 Sequence Data

### Examples

| Domain | Input | Output |
|--------|-------|--------|
| NLP | Word sequence | Sentiment/Translation |
| Time series | Historical values | Future prediction |
| Speech | Audio frames | Transcript |
| Video | Frame sequence | Action class |

### Challenges

- **Variable length**: Sequences have different lengths
- **Order matters**: "Dog bites man" ≠ "Man bites dog"
- **Long-range dependencies**: Context from distant past

### Why Not FC Networks?

- Fixed input size
- No parameter sharing across positions
- No notion of sequence order

---

## 11.2 Vanilla RNN

### Core Idea

Process sequence one element at a time, maintaining a **hidden state**.

### Equations

At time step $t$:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$y_t = W_y h_t + b_y$$

Where:
- $x_t$: Input at time $t$
- $h_t$: Hidden state at time $t$
- $y_t$: Output at time $t$
- $W_h, W_x, W_y$: Weight matrices (shared across time)

### Unrolled View

```
x₀   x₁   x₂   x₃
 ↓    ↓    ↓    ↓
[RNN]→[RNN]→[RNN]→[RNN]
 ↓    ↓    ↓    ↓
 y₀   y₁   y₂   y₃
```

### Parameter Sharing

Same weights $W_h, W_x, W_y$ at every time step → handles variable-length sequences.

### RNN Architectures

| Type | Description | Example |
|------|-------------|---------|
| Many-to-One | Sequence → Single output | Sentiment analysis |
| One-to-Many | Single input → Sequence | Image captioning |
| Many-to-Many | Sequence → Sequence | Translation |

---

## 11.3 Backpropagation Through Time (BPTT)

### Unrolling

Treat the unrolled RNN as a deep network.

### Gradient Computation

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_h}$$

Each $\frac{\partial L_t}{\partial W_h}$ involves chain rule through all previous steps:

$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

### The Gradient Problem

$$\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} W_h^T \cdot \text{diag}(\tanh'(z_i))$$

- If eigenvalues of $W_h$ < 1: **Vanishing gradients**
- If eigenvalues of $W_h$ > 1: **Exploding gradients**

### Solutions

| Problem | Solution |
|---------|----------|
| Exploding | Gradient clipping |
| Vanishing | LSTM, GRU architectures |

---

## 11.4 Long Short-Term Memory (LSTM)

### Key Innovation

Explicit **memory cell** with gating mechanisms to control information flow.

### Gates

| Gate | Purpose | Formula |
|------|---------|---------|
| Forget | What to erase | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ |
| Input | What to write | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ |
| Output | What to read | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ |

### Cell State Update

**Candidate cell state**:
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Cell state update**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Hidden state**:
$$h_t = o_t \odot \tanh(c_t)$$

### Why LSTM Works

- **Cell state** provides direct path for gradients (like skip connections)
- **Forget gate** can be near 1 → gradient flows through
- **Additive update** instead of multiplicative

### LSTM Diagram

```
           c_{t-1} ─────────[×]────────[+]─────── c_t
                              ↑          ↑
                           f_t       i_t⊙c̃_t
                              ↑          ↑
        x_t ─┬─→ [σ] f_t     [σ] i_t   [tanh] c̃_t
             │      ↑          ↑          ↑
        h_{t-1}─────┴──────────┴──────────┘
                              │
                           [σ] o_t
                              ↓
                          o_t⊙tanh(c_t) → h_t
```

---

## 11.5 Gated Recurrent Unit (GRU)

### Simplified Gating

Fewer parameters than LSTM, similar performance.

### Gates

| Gate | Purpose | Formula |
|------|---------|---------|
| Reset | How much past to forget | $r_t = \sigma(W_r [h_{t-1}, x_t])$ |
| Update | Interpolation factor | $z_t = \sigma(W_z [h_{t-1}, x_t])$ |

### State Update

**Candidate state**:
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$

**Hidden state** (interpolation):
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| State | Hidden + Cell | Hidden only |
| Parameters | More | Fewer |
| Performance | Slightly better | Similar |
| Training | Slower | Faster |

---

## 11.6 Bidirectional RNNs

### Motivation

Some tasks need future context (e.g., "I went to the [BLANK] to buy groceries").

### Architecture

Two RNNs: Forward and backward.

$$\overrightarrow{h}_t = \text{RNN}(\overrightarrow{h}_{t-1}, x_t)$$
$$\overleftarrow{h}_t = \text{RNN}(\overleftarrow{h}_{t+1}, x_t)$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

### Use Cases

- Named entity recognition
- Part-of-speech tagging
- Machine reading comprehension

---

## 11.7 Sequence-to-Sequence (Seq2Seq)

### Architecture

**Encoder**: Process input sequence → context vector
**Decoder**: Generate output sequence from context

```
Encoder:
x₁ → x₂ → x₃ → [h_T] (context)

Decoder:
[h_T] → y₁ → y₂ → y₃
```

### Training: Teacher Forcing

Feed ground-truth previous token (not prediction) during training.

### Inference: Autoregressive

Feed previous prediction as input.

### Applications

- Machine translation
- Text summarization
- Chatbots/dialogue

---

## 11.8 Implementation

### Vanilla RNN

```python
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
        self.hidden_size = hidden_size
    
    def forward(self, inputs, h_prev=None):
        """
        inputs: list of input vectors
        """
        if h_prev is None:
            h_prev = np.zeros(self.hidden_size)
        
        self.inputs = inputs
        self.hs = [h_prev]
        self.ys = []
        
        for x in inputs:
            h = np.tanh(x @ self.Wx + self.hs[-1] @ self.Wh + self.bh)
            y = h @ self.Wy + self.by
            self.hs.append(h)
            self.ys.append(y)
        
        return self.ys, self.hs[-1]
```

### LSTM Cell

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Combined weights for efficiency
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([h_prev, x])
        
        # Gates
        f = sigmoid(concat @ self.Wf + self.bf)
        i = sigmoid(concat @ self.Wi + self.bi)
        o = sigmoid(concat @ self.Wo + self.bo)
        
        # Cell state
        c_tilde = np.tanh(concat @ self.Wc + self.bc)
        c = f * c_prev + i * c_tilde
        
        # Hidden state
        h = o * np.tanh(c)
        
        return h, c

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

---

## 11.9 Practical Considerations

### Sequence Padding and Packing

- Pad shorter sequences to max length
- Use masks to ignore padded positions

### Gradient Clipping

```python
def clip_gradients(grads, max_norm=5.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = [g * clip_coef for g in grads]
    return grads
```

### Stacked/Deep RNNs

Multiple RNN layers stacked vertically for more capacity.

---

## Exercises

### Exercise 11.1 (Section 11.2)
Implement a character-level RNN that generates text.

### Exercise 11.2 (Section 11.3)
Demonstrate vanishing gradients in vanilla RNN on a long sequence task.

### Exercise 11.3 (Section 11.4)
Compare LSTM vs vanilla RNN on a task requiring long-term memory.

### Exercise 11.4 (Section 11.7)
Implement a seq2seq model for simple sequence reversal.

---

## Summary

- RNNs process sequences via hidden state
- Vanilla RNNs suffer from vanishing/exploding gradients
- LSTM uses gates and cell state for long-term dependencies
- GRU is a simplified alternative to LSTM
- Seq2seq enables sequence-to-sequence tasks

---

## References

### Primary Text
- Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 10.

### Key Papers
- Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
- Cho, K., et al. (2014). "GRU: Learning Phrase Representations."
- Sutskever, I., et al. (2014). "Sequence to Sequence Learning."

---

## Course Progression

Sequence models lead to:
- **Week 12**: Advanced seq2seq applications
- **Week 13**: Attention mechanisms that overcome RNN limitations
- **Week 13**: Transformers as the modern alternative to RNNs
- **NLP projects**: Text classification, generation, translation

