# IME 775 - Assignment 4


## Part A: Perceptrons and Decision Boundaries 

### Problem 1: Perceptron Computation

A single perceptron has weights $\vec{w} = [0.6, -0.4, 0.3]^T$ and bias $b = -0.1$, using the Heaviside activation $\theta(x)$.

**(a)** Compute the output for inputs $\vec{x}_1 = [1, 0, 1]^T$, $\vec{x}_2 = [0, 1, 1]^T$, and $\vec{x}_3 = [1, 1, 0]^T$.

**(b)** Write the equation of the decision boundary (hyperplane) for this perceptron.

**(c)** Is the point $\vec{x}_4 = [0.5, 0.5, 0.5]^T$ on the positive or negative side of the decision boundary?

---

### Problem 2: Designing Logic Gates

**(a)** Design a perceptron (find $w_1, w_2, b$) that implements the NAND gate:

| $x_1$ | $x_2$ | NAND |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**(b)** Verify your design by computing the output for all four input combinations.

**(c)** Explain why NAND is called a "universal gate" — how can you construct AND, OR, and NOT using only NAND gates?

---

### Problem 3: XOR and the Limits of Single Perceptrons

**(a)** Plot the four XOR input-output pairs on a 2D plane with class 0 as circles and class 1 as crosses. Explain geometrically why no single line can separate the two classes.

**(b)** Show algebraically that no single perceptron with Heaviside activation can compute XOR. That is, prove there are no values of $w_1, w_2, b$ satisfying all four constraints.

**(c)** Using the MLP decomposition $\text{XOR}(x_1, x_2) = \text{AND}(\text{OR}(x_1, x_2),\ \text{NAND}(x_1, x_2))$, design a 2-layer MLP (with explicit weights and biases for each neuron) that computes XOR. Verify for all four inputs.

---

## Part B: Activation Functions

### Problem 4: Sigmoid Computation and Properties

**(a)** Compute $\sigma(0)$, $\sigma(1)$, $\sigma(-2)$, and $\sigma(5)$.

**(b)** Prove the symmetry property: $\sigma(-x) = 1 - \sigma(x)$.

**(c)** Compute $\sigma'(0)$, $\sigma'(1)$, and $\sigma'(-2)$ using the formula $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

**(d)** At what value of $x$ is $\sigma'(x)$ maximized? What is the maximum value? Explain why this maximum being $< 1$ causes problems in deep networks.

---

### Problem 5: Tanh vs. Sigmoid

**(a)** Compute $\tanh(0)$, $\tanh(1)$, and $\tanh(-1)$.

**(b)** Verify numerically that $\tanh(x) = 2\sigma(2x) - 1$ for $x = 1$.

**(c)** Compute $\tanh'(0)$ and $\sigma'(0)$. By what factor is the tanh gradient stronger at $x = 0$?

**(d)** Explain why centered outputs (range $[-1, 1]$ for tanh vs. $(0, 1)$ for sigmoid) are beneficial for training. Consider what happens to the sign of gradients when all activations are positive.

---

### Problem 6: Parametrized Sigmoid

Consider the parametrized sigmoid $\sigma(x; w, b) = \frac{1}{1 + e^{-(wx+b)}}$.

**(a)** For $w = 1, b = 0$ and $w = 10, b = 0$, compute the output at $x = 0.1, 0.5, 0.9$. Which setting produces outputs closer to the Heaviside step function?

**(b)** For fixed $w = 5$, find the value of $b$ that shifts the transition point (where the output is $0.5$) to $x = 2$.

**(c)** Show that the derivative of the parametrized sigmoid is $\frac{d}{dx}\sigma(x; w, b) = w \cdot \sigma(x; w, b)(1 - \sigma(x; w, b))$.

---

## Part C: Forward Propagation 

### Problem 7: Single Layer Forward Pass

A linear layer has:

$$W = \begin{pmatrix} 0.3 & -0.5 & 0.2 \\ 0.1 & 0.4 & -0.3 \\ -0.2 & 0.6 & 0.1 \end{pmatrix}, \quad \vec{b} = \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix}$$

**(a)** Compute $\vec{z} = W\vec{a} + \vec{b}$ for input $\vec{a} = [1, 0.5, -1]^T$.

**(b)** Apply sigmoid activation: compute $\vec{a}_{\text{out}} = \sigma(\vec{z})$ elementwise.

**(c)** How many trainable parameters (weights + biases) does this layer have?

---

### Problem 8: Full MLP Forward Pass

Consider a 2 → 3 → 2 → 1 MLP with sigmoid activations throughout.

**Layer 0 (2 → 3):**
$$W^{(0)} = \begin{pmatrix} 0.5 & 0.3 \\ -0.2 & 0.4 \\ 0.1 & -0.6 \end{pmatrix}, \quad \vec{b}^{(0)} = \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix}$$

**Layer 1 (3 → 2):**
$$W^{(1)} = \begin{pmatrix} 0.4 & -0.3 & 0.2 \\ 0.1 & 0.5 & -0.4 \end{pmatrix}, \quad \vec{b}^{(1)} = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}$$

**Layer 2 (2 → 1):**
$$\vec{w}^{(2)} = \begin{pmatrix} 0.6 & -0.5 \end{pmatrix}, \quad b^{(2)} = 0.2$$

**(a)** Trace the forward pass for input $\vec{x} = [1, 0]^T$. Show $\vec{z}^{(l)}$ and $\vec{a}^{(l)}$ for each layer.

**(b)** Compute the total number of parameters (weights + biases) in this network.

**(c)** If the target output is $\bar{y} = 1$, compute the single-example MSE loss.

---

### Problem 9: Forward Propagation in PyTorch

**(a)** Write PyTorch code to define the exact network from Problem 8 (2 → 3 → 2 → 1 with sigmoid) using `nn.Sequential`. Initialize the weights and biases to the values given in Problem 8 using `torch.nn.Parameter` or direct assignment.

**(b)** Use your model to compute the forward pass for $\vec{x} = [1, 0]^T$ and verify that the output matches your hand calculation from Problem 8(a).

**(c)** How can you access the weight matrix of the first layer after defining the model? Write the code.

---

## Part D: Loss and Backpropagation

### Problem 10: MSE Loss Computation

A network with 2 outputs is evaluated on 3 training examples:

| Example | Target $\bar{y}$ | Predicted $y$ |
|---|---|---|
| 1 | $(1.0, 0.0)$ | $(0.8, 0.2)$ |
| 2 | $(0.0, 1.0)$ | $(0.1, 0.7)$ |
| 3 | $(1.0, 1.0)$ | $(0.6, 0.9)$ |

**(a)** Compute the per-example loss $\ell_i$ for each example.

**(b)** Compute the total MSE loss $L = \sum_{i=1}^{3} \ell_i$.

**(c)** Which training example contributes the most to the total loss? What does this suggest about where the network needs to improve?

---

### Problem 11: Backpropagation on a Simple Network

Consider a 1 → 1 → 1 network (one neuron per layer) with sigmoid activation:

- Layer 0: $w^{(0)} = 0.5$, $b^{(0)} = 0.1$
- Layer 1: $w^{(1)} = 0.8$, $b^{(1)} = -0.2$
- Input: $x = 1.0$, Target: $\bar{y} = 1.0$

**(a)** Perform the forward pass. Compute $z^{(0)}$, $a^{(0)}$, $z^{(1)}$, $a^{(1)} = y$.

**(b)** Compute the loss $\ell = \frac{1}{2}(\bar{y} - y)^2$.

**(c)** Compute $\delta^{(1)}$ for the output layer.

**(d)** Compute all four gradients: $\frac{\partial \ell}{\partial w^{(1)}}$, $\frac{\partial \ell}{\partial b^{(1)}}$, $\frac{\partial \ell}{\partial w^{(0)}}$, $\frac{\partial \ell}{\partial b^{(0)}}$.

**(e)** With learning rate $r = 0.5$, compute the updated values of all four parameters.

---

### Problem 12: Backpropagation on a General Network

Using the 1 → 2 → 1 network from the Lecture 16 workout:

$$W^{(0)} = \begin{pmatrix} 0.5 \\ -0.3 \end{pmatrix}, \quad \vec{b}^{(0)} = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}, \quad \vec{w}^{(1)} = \begin{pmatrix} 0.4 & 0.6 \end{pmatrix}, \quad b^{(1)} = 0.2$$

Input $x = 1.0$, target $\bar{y} = 0.0$, sigmoid activation.

**(a)** Perform the forward pass (compute all $z$ and $a$ values).

**(b)** Compute $\delta^{(1)}$ and the gradients for layer 1.

**(c)** Compute $\vec{\delta}^{(0)}$ using the backpropagation recursion and the gradients for layer 0.

**(d)** Update all weights and biases with learning rate $r = 1.0$.

**(e)** Perform a second forward pass with the updated parameters. Is the new prediction closer to the target $\bar{y} = 0$?

---

## Part E: Conceptual and PyTorch

### Problem 13: True or False

State whether each statement is True or False and provide a brief justification.

**(a)** A single perceptron with Heaviside activation can learn any Boolean function of two inputs.

**(b)** The sigmoid function always produces outputs in the range $[0, 1]$ (inclusive of both endpoints).

**(c)** During backpropagation, we need to store all pre-activation vectors $\vec{z}^{(l)}$ from the forward pass.

**(d)** If the learning rate is too large, gradient descent can cause the loss to increase.

**(e)** In a network with $L$ sigmoid layers, the gradient at the first layer is always smaller than the gradient at the last layer.

**(f)** The `zero_grad()` call in PyTorch is optional and only affects computational efficiency.

---

### Problem 14: Connecting the Chapters

Trace the logical progression from Chapter 7 to Chapter 8 by answering:

**(a)** Why is the Heaviside step function unsuitable for training neural networks? Give a mathematical reason.

**(b)** Why is forward propagation a prerequisite for backpropagation? What values computed during the forward pass are needed during the backward pass?

**(c)** Explain the relationship between the chain rule of calculus and the backpropagation delta recursion $\vec{\delta}^{(l)} = [(W^{(l+1)})^T \vec{\delta}^{(l+1)}] \odot f'(\vec{z}^{(l)})$.

**(d)** If we used Heaviside activations in backpropagation, what would $\delta^{(l)}$ be for most inputs? Why does this make learning impossible?

---

### Problem 15: PyTorch Training

**(a)** Write a complete PyTorch training loop to train a 2 → 4 → 4 → 1 network with sigmoid activations to solve the XOR problem. Use MSE loss and SGD optimizer with learning rate 1.0. Train for 10,000 epochs.

**(b)** After training, print the network's predictions for all four XOR inputs. Does the network converge to the correct outputs?

**(c)** Modify the code to use `tanh` activations instead of sigmoid. Does the network converge faster or slower? Report the loss at epoch 2,000 for both activation functions.

**(d)** Explain why the learning rate that works well for sigmoid might not work well for tanh (hint: consider the gradient magnitudes).
