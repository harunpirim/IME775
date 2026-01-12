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
    # Week 11: Recurrent Neural Networks (RNNs)

    **IME775: Data Driven Modeling and Optimization**

    📖 **Reference**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 10

    ---

    ## Learning Objectives

    - Understand sequence modeling challenges
    - Master RNN architecture and mathematics
    - Learn LSTM and GRU mechanisms
    - Implement basic sequence models
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
    ## 11.1 Why Recurrent Networks?

    **Sequential data challenges:**
    - Variable length sequences
    - Order matters
    - Long-range dependencies

    **Solution**: Process one element at a time, maintain **hidden state**
    """)
    return


@app.cell
def _(np, plt):
    # Visualize RNN unrolling
    fig1, ax1 = plt.subplots(figsize=(14, 5))

    # Draw unrolled RNN
    n_steps = 5
    
    for t in range(n_steps):
        x_pos = t * 2.5
        
        # Input
        ax1.add_patch(plt.Circle((x_pos, 0), 0.3, fill=True, facecolor='lightblue', edgecolor='black'))
        ax1.text(x_pos, 0, f'x{t}', ha='center', va='center', fontsize=10)
        
        # RNN cell
        ax1.add_patch(plt.Rectangle((x_pos - 0.4, 1.2), 0.8, 0.8, fill=True, 
                                      facecolor='lightgreen', edgecolor='black'))
        ax1.text(x_pos, 1.6, 'RNN', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Output
        ax1.add_patch(plt.Circle((x_pos, 3), 0.3, fill=True, facecolor='lightyellow', edgecolor='black'))
        ax1.text(x_pos, 3, f'y{t}', ha='center', va='center', fontsize=10)
        
        # Input to cell arrow
        ax1.annotate('', xy=(x_pos, 1.2), xytext=(x_pos, 0.3),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        # Cell to output arrow
        ax1.annotate('', xy=(x_pos, 2.7), xytext=(x_pos, 2),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        # Recurrent connection
        if t < n_steps - 1:
            ax1.annotate('', xy=(x_pos + 2.1, 1.6), xytext=(x_pos + 0.4, 1.6),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax1.text(x_pos + 1.25, 1.85, f'h{t}', ha='center', fontsize=9, color='red')

    # Initial hidden state
    ax1.annotate('', xy=(-0.4, 1.6), xytext=(-1.2, 1.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(-1.5, 1.6, 'h₀', ha='center', fontsize=10, color='red')

    ax1.set_xlim(-2, 12)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('RNN Unrolled Through Time\n(Same weights at each step)', fontsize=14)

    fig1
    return ax1, fig1, n_steps, t, x_pos


@app.cell
def _(mo):
    mo.md(r"""
    ## 11.2 RNN Equations

    At each time step $t$:

    $$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
    $$y_t = W_y h_t + b_y$$

    **Key**: Same weights $W_h, W_x, W_y$ at every time step!
    """)
    return


@app.cell
def _(np, plt):
    # Implement and visualize simple RNN
    class SimpleRNN:
        def __init__(self, input_size, hidden_size):
            np.random.seed(42)
            self.Wx = np.random.randn(input_size, hidden_size) * 0.1
            self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
            self.bh = np.zeros(hidden_size)
            self.hidden_size = hidden_size
        
        def forward(self, inputs):
            """inputs: (seq_len, input_size)"""
            h = np.zeros(self.hidden_size)
            hidden_states = [h]
            
            for x in inputs:
                h = np.tanh(x @ self.Wx + h @ self.Wh + self.bh)
                hidden_states.append(h)
            
            return np.array(hidden_states[1:])

    # Generate a simple sine wave input
    t_seq = np.linspace(0, 4*np.pi, 50)
    x_seq = np.sin(t_seq).reshape(-1, 1)

    # Process with RNN
    rnn = SimpleRNN(input_size=1, hidden_size=8)
    h_seq = rnn.forward(x_seq)

    # Visualize
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6))

    axes2[0].plot(t_seq, x_seq, 'b-', linewidth=2)
    axes2[0].set_xlabel('Time')
    axes2[0].set_ylabel('Input x(t)')
    axes2[0].set_title('Input Sequence: Sine Wave')
    axes2[0].grid(True, alpha=0.3)

    # Plot hidden state activations
    im = axes2[1].imshow(h_seq.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes2[1].set_xlabel('Time Step')
    axes2[1].set_ylabel('Hidden Unit')
    axes2[1].set_title('Hidden State Evolution Over Time')
    plt.colorbar(im, ax=axes2[1], label='Activation')

    plt.tight_layout()
    fig2
    return SimpleRNN, axes2, fig2, h_seq, im, rnn, t_seq, x_seq


@app.cell
def _(mo):
    mo.md(r"""
    ## 11.3 The Vanishing Gradient Problem

    For long sequences, gradients must flow through many time steps:

    $$\frac{\partial L_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

    If each factor < 1 → **Vanishing gradients**
    If each factor > 1 → **Exploding gradients**
    """)
    return


@app.cell
def _(np, plt):
    # Demonstrate vanishing gradients
    def simulate_gradient_flow_rnn(seq_len, weight_scale=0.9):
        gradients = [1.0]
        
        for t in range(seq_len):
            # Approximate gradient factor (simplified)
            # In reality: W_h^T * tanh'(z)
            # tanh' is between 0 and 1, so gradient shrinks
            grad_factor = weight_scale * np.random.uniform(0.8, 1.0)
            gradients.append(gradients[-1] * grad_factor)
        
        return gradients

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    seq_lengths = [50, 100, 200]

    # Vanishing gradients
    for seq_len in seq_lengths:
        grads = simulate_gradient_flow_rnn(seq_len, weight_scale=0.9)
        axes3[0].semilogy(grads, label=f'T={seq_len}', linewidth=2)

    axes3[0].axhline(1e-10, color='gray', linestyle='--', label='Numerical limit')
    axes3[0].set_xlabel('Time Steps Back')
    axes3[0].set_ylabel('Gradient Magnitude (log)')
    axes3[0].set_title('Vanishing Gradients in RNN')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Solution comparison
    def lstm_gradient_flow(seq_len):
        # LSTM maintains gradient through cell state (close to 1)
        gradients = [1.0]
        for t in range(seq_len):
            # Forget gate can be close to 1
            forget_gate = np.random.uniform(0.9, 1.0)
            gradients.append(gradients[-1] * forget_gate)
        return gradients

    rnn_grads = simulate_gradient_flow_rnn(100, 0.9)
    lstm_grads = lstm_gradient_flow(100)

    axes3[1].semilogy(rnn_grads, 'r-', linewidth=2, label='Vanilla RNN')
    axes3[1].semilogy(lstm_grads, 'g-', linewidth=2, label='LSTM')
    axes3[1].axhline(1e-10, color='gray', linestyle='--')
    axes3[1].set_xlabel('Time Steps Back')
    axes3[1].set_ylabel('Gradient Magnitude (log)')
    axes3[1].set_title('LSTM vs RNN: Gradient Flow')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3
    return (
        axes3,
        fig3,
        grads,
        lstm_gradient_flow,
        lstm_grads,
        rnn_grads,
        seq_len,
        seq_lengths,
        simulate_gradient_flow_rnn,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 11.4 Long Short-Term Memory (LSTM)

    **Key Innovation**: Explicit memory cell with gating

    | Gate | Purpose |
    |------|---------|
    | Forget ($f_t$) | What to erase from memory |
    | Input ($i_t$) | What to write to memory |
    | Output ($o_t$) | What to read from memory |

    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
    $$h_t = o_t \odot \tanh(c_t)$$
    """)
    return


@app.cell
def _(np, plt):
    # Visualize LSTM gates
    fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4))

    # Simulate gate activations over time
    np.random.seed(42)
    T = 30

    # Create a scenario: remember information at t=5, forget at t=20
    t_range = np.arange(T)

    # Forget gate (starts high, drops at t=20)
    forget_gate = np.ones(T) * 0.95
    forget_gate[20:25] = np.linspace(0.95, 0.1, 5)
    forget_gate[25:] = 0.1
    forget_gate += np.random.randn(T) * 0.03

    # Input gate (spikes at t=5)
    input_gate = np.ones(T) * 0.1
    input_gate[5:10] = np.linspace(0.1, 0.9, 5)
    input_gate[10:15] = np.linspace(0.9, 0.1, 5)
    input_gate += np.random.randn(T) * 0.03

    # Output gate
    output_gate = np.ones(T) * 0.5 + np.random.randn(T) * 0.1

    # Cell state (accumulates based on gates)
    cell_state = np.zeros(T)
    c = 0
    for t in range(T):
        c = np.clip(forget_gate[t], 0, 1) * c + np.clip(input_gate[t], 0, 1) * 0.5
        cell_state[t] = c

    axes4[0].plot(t_range, np.clip(forget_gate, 0, 1), 'r-', linewidth=2)
    axes4[0].fill_between(t_range, np.clip(forget_gate, 0, 1), alpha=0.3, color='red')
    axes4[0].set_title('Forget Gate $f_t$\n(What to keep)')
    axes4[0].set_xlabel('Time')
    axes4[0].set_ylim(0, 1)
    axes4[0].axvline(20, color='gray', linestyle='--', alpha=0.5)
    axes4[0].grid(True, alpha=0.3)

    axes4[1].plot(t_range, np.clip(input_gate, 0, 1), 'g-', linewidth=2)
    axes4[1].fill_between(t_range, np.clip(input_gate, 0, 1), alpha=0.3, color='green')
    axes4[1].set_title('Input Gate $i_t$\n(What to write)')
    axes4[1].set_xlabel('Time')
    axes4[1].set_ylim(0, 1)
    axes4[1].axvline(5, color='gray', linestyle='--', alpha=0.5)
    axes4[1].grid(True, alpha=0.3)

    axes4[2].plot(t_range, np.clip(output_gate, 0, 1), 'b-', linewidth=2)
    axes4[2].fill_between(t_range, np.clip(output_gate, 0, 1), alpha=0.3, color='blue')
    axes4[2].set_title('Output Gate $o_t$\n(What to read)')
    axes4[2].set_xlabel('Time')
    axes4[2].set_ylim(0, 1)
    axes4[2].grid(True, alpha=0.3)

    axes4[3].plot(t_range, cell_state, 'm-', linewidth=2)
    axes4[3].fill_between(t_range, cell_state, alpha=0.3, color='purple')
    axes4[3].set_title('Cell State $c_t$\n(Memory)')
    axes4[3].set_xlabel('Time')
    axes4[3].axvline(5, color='green', linestyle='--', alpha=0.5, label='Write')
    axes4[3].axvline(20, color='red', linestyle='--', alpha=0.5, label='Forget')
    axes4[3].legend()
    axes4[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig4
    return (
        T,
        axes4,
        c,
        cell_state,
        fig4,
        forget_gate,
        input_gate,
        output_gate,
        t,
        t_range,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 11.5 GRU: Simplified Gating

    **Gated Recurrent Unit**: Fewer parameters, similar performance

    | Gate | Purpose |
    |------|---------|
    | Reset ($r_t$) | How much past to forget |
    | Update ($z_t$) | Interpolation factor |

    $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
    """)
    return


@app.cell
def _(np, plt):
    # Compare architectures
    fig5, ax5 = plt.subplots(figsize=(12, 6))

    architectures = {
        'Vanilla RNN': {'params': 1, 'states': 1, 'gates': 0},
        'GRU': {'params': 3, 'states': 1, 'gates': 2},
        'LSTM': {'params': 4, 'states': 2, 'gates': 3},
    }

    x_arch = np.arange(len(architectures))
    width = 0.25

    params = [v['params'] for v in architectures.values()]
    states = [v['states'] for v in architectures.values()]
    gates = [v['gates'] for v in architectures.values()]

    ax5.bar(x_arch - width, params, width, label='Weight Matrices (×hidden²)', color='blue', alpha=0.7)
    ax5.bar(x_arch, states, width, label='State Vectors', color='green', alpha=0.7)
    ax5.bar(x_arch + width, gates, width, label='Gates', color='red', alpha=0.7)

    ax5.set_xticks(x_arch)
    ax5.set_xticklabels(architectures.keys())
    ax5.set_ylabel('Count')
    ax5.set_title('Architecture Comparison: Complexity vs Capability')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Add annotations
    ax5.annotate('Simple but\nvanishing gradients', xy=(0, 1), xytext=(0, 1.5),
                fontsize=10, ha='center')
    ax5.annotate('Good tradeoff\nfewer params', xy=(1, 3), xytext=(1, 3.5),
                fontsize=10, ha='center')
    ax5.annotate('Most powerful\nmore params', xy=(2, 4), xytext=(2, 4.5),
                fontsize=10, ha='center')

    fig5
    return architectures, ax5, fig5, gates, params, states, width, x_arch


@app.cell
def _(mo):
    mo.md(r"""
    ## 11.6 Sequence-to-Sequence (Seq2Seq)

    **Encoder**: Process input → context vector
    **Decoder**: Generate output from context

    ```
    [Hello] [World] → Encoder → [context] → Decoder → [Bonjour] [Monde]
    ```
    """)
    return


@app.cell
def _(np, plt):
    # Visualize seq2seq
    fig6, ax6 = plt.subplots(figsize=(14, 6))

    # Encoder
    encoder_words = ['Hello', 'World', '<EOS>']
    for i, word in enumerate(encoder_words):
        x_enc = i * 1.5
        ax6.add_patch(plt.Circle((x_enc, 0), 0.3, fill=True, facecolor='lightblue', edgecolor='black'))
        ax6.text(x_enc, 0, word, ha='center', va='center', fontsize=8)
        
        ax6.add_patch(plt.Rectangle((x_enc - 0.35, 1), 0.7, 0.7, fill=True, 
                                     facecolor='lightgreen', edgecolor='black'))
        ax6.text(x_enc, 1.35, 'Enc', ha='center', va='center', fontsize=9)
        
        ax6.annotate('', xy=(x_enc, 1), xytext=(x_enc, 0.3),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        if i < len(encoder_words) - 1:
            ax6.annotate('', xy=(x_enc + 1.15, 1.35), xytext=(x_enc + 0.35, 1.35),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Context vector
    ax6.add_patch(plt.Circle((4.5, 1.35), 0.4, fill=True, facecolor='yellow', edgecolor='black'))
    ax6.text(4.5, 1.35, 'ctx', ha='center', va='center', fontsize=10, fontweight='bold')
    ax6.annotate('', xy=(4.1, 1.35), xytext=(3.35, 1.35),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Decoder
    decoder_words = ['<SOS>', 'Bonjour', 'Monde']
    for i, word in enumerate(decoder_words):
        x_dec = 6 + i * 1.5
        
        ax6.add_patch(plt.Rectangle((x_dec - 0.35, 1), 0.7, 0.7, fill=True, 
                                     facecolor='lightyellow', edgecolor='black'))
        ax6.text(x_dec, 1.35, 'Dec', ha='center', va='center', fontsize=9)
        
        if i < len(decoder_words) - 1:
            ax6.add_patch(plt.Circle((x_dec, 2.5), 0.3, fill=True, facecolor='lightcoral', edgecolor='black'))
            ax6.text(x_dec, 2.5, decoder_words[i+1], ha='center', va='center', fontsize=8)
            ax6.annotate('', xy=(x_dec, 2.2), xytext=(x_dec, 1.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
        
        if i < len(decoder_words) - 1:
            ax6.annotate('', xy=(x_dec + 1.15, 1.35), xytext=(x_dec + 0.35, 1.35),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Context to decoder
    ax6.annotate('', xy=(5.65, 1.35), xytext=(4.9, 1.35),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))

    # Labels
    ax6.text(1.5, -0.8, 'ENCODER', ha='center', fontsize=12, fontweight='bold', color='green')
    ax6.text(7.5, -0.8, 'DECODER', ha='center', fontsize=12, fontweight='bold', color='orange')

    ax6.set_xlim(-1, 11)
    ax6.set_ylim(-1.2, 3)
    ax6.axis('off')
    ax6.set_title('Sequence-to-Sequence Architecture\n(Machine Translation Example)', fontsize=14)

    fig6
    return ax6, decoder_words, encoder_words, fig6, i, word, x_dec, x_enc


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Architecture | Key Feature | Use Case |
    |--------------|-------------|----------|
    | **Vanilla RNN** | Simple hidden state | Short sequences |
    | **LSTM** | Cell state + 3 gates | Long-term dependencies |
    | **GRU** | 2 gates, simpler | Similar to LSTM |
    | **Bidirectional** | Forward + backward | Full context needed |
    | **Seq2Seq** | Encoder-decoder | Translation, summarization |

    ---

    ## References

    - **Primary**: Krishnendu Chaudhury. *Math and Architectures of Deep Learning*, Chapter 10.
    - **LSTM**: Hochreiter & Schmidhuber (1997)
    - **GRU**: Cho et al. (2014)

    ## Connection to ML Refined Curriculum

    RNNs extend time series from Week 10:
    - Automatic feature learning from sequences
    - Handle variable-length data
    """)
    return


if __name__ == "__main__":
    app.run()

