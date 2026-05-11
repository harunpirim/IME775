# IME 775: Problem Set 5 — Solutions

## Chapters 7 & 8

---

## Part A: Perceptrons and Decision Boundaries

### Problem 1: Perceptron Computation

**(a)** Output = $\theta(\vec{w} \cdot \vec{x} + b)$

- $\vec{x}_1 = [1,0,1]^T$: $z = 0.6(1) + (-0.4)(0) + 0.3(1) - 0.1 = 0.6 + 0 + 0.3 - 0.1 = 0.8$. Output = $\theta(0.8) = 1$
- $\vec{x}_2 = [0,1,1]^T$: $z = 0.6(0) + (-0.4)(1) + 0.3(1) - 0.1 = 0 - 0.4 + 0.3 - 0.1 = -0.2$. Output = $\theta(-0.2) = 0$
- $\vec{x}_3 = [1,1,0]^T$: $z = 0.6(1) + (-0.4)(1) + 0.3(0) - 0.1 = 0.6 - 0.4 + 0 - 0.1 = 0.1$. Output = $\theta(0.1) = 1$

**(b)** The decision boundary is $\vec{w} \cdot \vec{x} + b = 0$:

$$0.6x_1 - 0.4x_2 + 0.3x_3 - 0.1 = 0$$

**(c)** $z = 0.6(0.5) - 0.4(0.5) + 0.3(0.5) - 0.1 = 0.3 - 0.2 + 0.15 - 0.1 = 0.15 > 0$

Since $z > 0$, $\vec{x}_4$ is on the **positive side** (output = 1).

---

### Problem 2: Designing Logic Gates

**(a)** NAND gate design: $w_1 = -1, w_2 = -1, b = 1.5$

**(b)** Verification:
- $(0,0)$: $z = -1(0) - 1(0) + 1.5 = 1.5 > 0 \Rightarrow \theta = 1$ ✓
- $(0,1)$: $z = -1(0) - 1(1) + 1.5 = 0.5 > 0 \Rightarrow \theta = 1$ ✓
- $(1,0)$: $z = -1(1) - 1(0) + 1.5 = 0.5 > 0 \Rightarrow \theta = 1$ ✓
- $(1,1)$: $z = -1(1) - 1(1) + 1.5 = -0.5 < 0 \Rightarrow \theta = 0$ ✓

**(c)** NAND is universal because:
- **NOT**(A) = NAND(A, A)
- **AND**(A, B) = NOT(NAND(A, B)) = NAND(NAND(A, B), NAND(A, B))
- **OR**(A, B) = NAND(NOT(A), NOT(B)) = NAND(NAND(A, A), NAND(B, B))

Any Boolean function can be expressed using combinations of AND, OR, NOT, so NAND alone suffices.

---

### Problem 3: XOR and the Limits of Single Perceptrons

**(a)** Plotting the points:
- Class 0: $(0,0)$ and $(1,1)$ — on one diagonal
- Class 1: $(0,1)$ and $(1,0)$ — on the other diagonal

No single line can separate the two classes because the positive examples are on opposite corners of the unit square, and the negative examples are on the other two corners. Any line separating $(0,1)$ from $(0,0)$ would place $(1,0)$ and $(1,1)$ on the same side.

**(b)** For a single perceptron, we need:

$$w_1(0) + w_2(0) + b < 0 \quad \Rightarrow \quad b < 0 \quad \text{...(i)}$$
$$w_1(0) + w_2(1) + b \geq 0 \quad \Rightarrow \quad w_2 + b \geq 0 \quad \text{...(ii)}$$
$$w_1(1) + w_2(0) + b \geq 0 \quad \Rightarrow \quad w_1 + b \geq 0 \quad \text{...(iii)}$$
$$w_1(1) + w_2(1) + b < 0 \quad \Rightarrow \quad w_1 + w_2 + b < 0 \quad \text{...(iv)}$$

From (ii) and (iii): $w_1 + w_2 + 2b \geq 0$, so $w_1 + w_2 \geq -2b > 0$ (using (i)).

But (iv) says $w_1 + w_2 < -b$, and from (i) $-b > 0$, so we need $w_1 + w_2 < -b$.

From (ii): $w_2 \geq -b > 0$ and from (iii): $w_1 \geq -b > 0$.

Adding: $w_1 + w_2 \geq -2b$.

But (iv) requires $w_1 + w_2 + b < 0$, i.e., $w_1 + w_2 < -b$.

Since $w_1 + w_2 \geq -2b$ and we need $w_1 + w_2 < -b$, this means $-2b \leq w_1 + w_2 < -b$, i.e., $-2b < -b$, i.e., $b > 0$. This contradicts (i). ∎

**(c)** MLP for XOR:

**Hidden layer (layer 0):**
- OR neuron: $w_{01} = 1, w_{02} = 1, b_0 = -0.5$ → outputs 1 if $x_1 + x_2 \geq 0.5$
- NAND neuron: $w_{01} = -1, w_{02} = -1, b_0 = 1.5$ → outputs 1 if $-x_1 - x_2 + 1.5 \geq 0$

$$W^{(0)} = \begin{pmatrix} 1 & 1 \\ -1 & -1 \end{pmatrix}, \quad \vec{b}^{(0)} = \begin{pmatrix} -0.5 \\ 1.5 \end{pmatrix}$$

**Output layer (layer 1):** AND of the two hidden neurons:

$$\vec{w}^{(1)} = \begin{pmatrix} 1 & 1 \end{pmatrix}, \quad b^{(1)} = -1.5$$

**Verification:**
- $(0,0)$: hidden = $[\theta(-0.5), \theta(1.5)] = [0, 1]$; output = $\theta(0 + 1 - 1.5) = \theta(-0.5) = 0$ ✓
- $(0,1)$: hidden = $[\theta(0.5), \theta(0.5)] = [1, 1]$; output = $\theta(1 + 1 - 1.5) = \theta(0.5) = 1$ ✓
- $(1,0)$: hidden = $[\theta(0.5), \theta(0.5)] = [1, 1]$; output = $\theta(1 + 1 - 1.5) = \theta(0.5) = 1$ ✓
- $(1,1)$: hidden = $[\theta(1.5), \theta(-0.5)] = [1, 0]$; output = $\theta(1 + 0 - 1.5) = \theta(-0.5) = 0$ ✓

---

## Part B: Activation Functions

### Problem 4: Sigmoid Computation and Properties

**(a)**
- $\sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5$
- $\sigma(1) = \frac{1}{1+e^{-1}} = \frac{1}{1+0.3679} = \frac{1}{1.3679} \approx 0.7311$
- $\sigma(-2) = \frac{1}{1+e^2} = \frac{1}{1+7.389} = \frac{1}{8.389} \approx 0.1192$
- $\sigma(5) = \frac{1}{1+e^{-5}} = \frac{1}{1+0.00674} \approx 0.9933$

**(b)** $\sigma(-x) = \frac{1}{1+e^{-(-x)}} = \frac{1}{1+e^x}$

$1 - \sigma(x) = 1 - \frac{1}{1+e^{-x}} = \frac{1+e^{-x}-1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{e^x + 1} = \frac{1}{1+e^x}$

Since both equal $\frac{1}{1+e^x}$, we have $\sigma(-x) = 1 - \sigma(x)$. ∎

**(c)**
- $\sigma'(0) = \sigma(0)(1-\sigma(0)) = 0.5 \times 0.5 = 0.25$
- $\sigma'(1) = \sigma(1)(1-\sigma(1)) = 0.7311 \times 0.2689 \approx 0.1966$
- $\sigma'(-2) = \sigma(-2)(1-\sigma(-2)) = 0.1192 \times 0.8808 \approx 0.1050$

**(d)** $\sigma'(x) = \sigma(x)(1-\sigma(x))$ is maximized when $\sigma(x) = 0.5$, i.e., at $x = 0$.

Maximum value: $0.5 \times 0.5 = 0.25$.

Since $\sigma'(x) \leq 0.25 < 1$, each layer in backpropagation multiplies gradients by a factor $\leq 0.25$. After $L$ layers, gradients shrink by up to $0.25^L$. For $L = 10$ layers: $0.25^{10} \approx 10^{-6}$. The first layers receive negligibly small gradients and barely learn — the **vanishing gradient problem**.

---

### Problem 5: Tanh vs. Sigmoid

**(a)**
- $\tanh(0) = \frac{e^0 - e^0}{e^0 + e^0} = \frac{0}{2} = 0$
- $\tanh(1) = \frac{e - e^{-1}}{e + e^{-1}} = \frac{2.718 - 0.368}{2.718 + 0.368} = \frac{2.350}{3.086} \approx 0.7616$
- $\tanh(-1) = -\tanh(1) \approx -0.7616$

**(b)** $2\sigma(2 \cdot 1) - 1 = 2\sigma(2) - 1$

$\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} \approx 0.8808$

$2(0.8808) - 1 = 1.7616 - 1 = 0.7616 = \tanh(1)$ ✓

**(c)**
- $\tanh'(0) = 1 - \tanh^2(0) = 1 - 0 = 1.0$
- $\sigma'(0) = 0.25$

Ratio: $\frac{\tanh'(0)}{\sigma'(0)} = \frac{1.0}{0.25} = 4$. Tanh gradient is **4 times stronger** at $x = 0$.

**(d)** When all activations are positive (as with sigmoid), the gradients $\frac{\partial L}{\partial w_{jk}} = \delta_j \cdot a_k$ have the **same sign** as $\delta_j$ for all $k$ (since $a_k > 0$ always). This forces all weights feeding into a neuron to update in the same direction, causing a "zig-zagging" optimization path. Tanh produces both positive and negative activations, allowing different weights to update in different directions, leading to more efficient optimization.

---

### Problem 6: Parametrized Sigmoid

**(a)**

For $w = 1, b = 0$: $\sigma(x; 1, 0) = \sigma(x)$
- $\sigma(0.1) \approx 0.525$
- $\sigma(0.5) \approx 0.622$
- $\sigma(0.9) \approx 0.711$

For $w = 10, b = 0$: $\sigma(x; 10, 0) = \sigma(10x)$
- $\sigma(1.0) \approx 0.731$
- $\sigma(5.0) \approx 0.993$
- $\sigma(9.0) \approx 0.9999$

$w = 10$ produces outputs **much closer to the Heaviside step function** — nearly 0 for $x < 0$ and nearly 1 for $x > 0$, with a very sharp transition at $x = 0$.

**(b)** The transition point is where the output is $0.5$, i.e., where $wx + b = 0$.

$$5(2) + b = 0 \Rightarrow b = -10$$

**(c)** Let $u = wx + b$, then $\sigma(x; w, b) = \sigma(u)$.

$$\frac{d}{dx}\sigma(u) = \sigma'(u) \cdot \frac{du}{dx} = \sigma(u)(1 - \sigma(u)) \cdot w = w \cdot \sigma(x; w, b)(1 - \sigma(x; w, b))$$ ∎

---

## Part C: Forward Propagation

### Problem 7: Single Layer Forward Pass

**(a)**

$$\vec{z} = \begin{pmatrix} 0.3 & -0.5 & 0.2 \\ 0.1 & 0.4 & -0.3 \\ -0.2 & 0.6 & 0.1 \end{pmatrix}\begin{pmatrix} 1 \\ 0.5 \\ -1 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix}$$

$$= \begin{pmatrix} 0.3 - 0.25 - 0.2 \\ 0.1 + 0.2 + 0.3 \\ -0.2 + 0.3 - 0.1 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix} = \begin{pmatrix} -0.15 \\ 0.6 \\ 0.0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix} = \begin{pmatrix} -0.05 \\ 0.5 \\ 0.2 \end{pmatrix}$$

**(b)**

$$\vec{a}_{\text{out}} = \begin{pmatrix} \sigma(-0.05) \\ \sigma(0.5) \\ \sigma(0.2) \end{pmatrix} \approx \begin{pmatrix} 0.4875 \\ 0.6225 \\ 0.5498 \end{pmatrix}$$

**(c)** Weights: $3 \times 3 = 9$. Biases: $3$. **Total: 12 parameters**.

---

### Problem 8: Full MLP Forward Pass

**(a)**

**Layer 0 (2 → 3):**

$$\vec{z}^{(0)} = \begin{pmatrix} 0.5 & 0.3 \\ -0.2 & 0.4 \\ 0.1 & -0.6 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix} = \begin{pmatrix} 0.5 \\ -0.2 \\ 0.1 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \\ 0.2 \end{pmatrix} = \begin{pmatrix} 0.6 \\ -0.3 \\ 0.3 \end{pmatrix}$$

$$\vec{a}^{(0)} = \begin{pmatrix} \sigma(0.6) \\ \sigma(-0.3) \\ \sigma(0.3) \end{pmatrix} \approx \begin{pmatrix} 0.6457 \\ 0.4256 \\ 0.5744 \end{pmatrix}$$

**Layer 1 (3 → 2):**

$$\vec{z}^{(1)} = \begin{pmatrix} 0.4 & -0.3 & 0.2 \\ 0.1 & 0.5 & -0.4 \end{pmatrix}\begin{pmatrix} 0.6457 \\ 0.4256 \\ 0.5744 \end{pmatrix} + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}$$

$$= \begin{pmatrix} 0.2583 - 0.1277 + 0.1149 \\ 0.0646 + 0.2128 - 0.2298 \end{pmatrix} + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 0.2455 \\ 0.0476 \end{pmatrix} + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 0.2455 \\ 0.1476 \end{pmatrix}$$

$$\vec{a}^{(1)} = \begin{pmatrix} \sigma(0.2455) \\ \sigma(0.1476) \end{pmatrix} \approx \begin{pmatrix} 0.5611 \\ 0.5368 \end{pmatrix}$$

**Layer 2 (2 → 1):**

$$z^{(2)} = 0.6(0.5611) + (-0.5)(0.5368) + 0.2 = 0.3367 - 0.2684 + 0.2 = 0.2683$$

$$y = a^{(2)} = \sigma(0.2683) \approx 0.5667$$

**(b)** Parameters:
- Layer 0: $3 \times 2 + 3 = 9$ (6 weights + 3 biases)
- Layer 1: $2 \times 3 + 2 = 8$ (6 weights + 2 biases)
- Layer 2: $1 \times 2 + 1 = 3$ (2 weights + 1 bias)
- **Total: 20 parameters**

**(c)** $\ell = \frac{1}{2}(1 - 0.5667)^2 = \frac{1}{2}(0.4333)^2 = \frac{1}{2}(0.1877) \approx 0.0939$

---

### Problem 9: Forward Propagation in PyTorch

**(a)**

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Sigmoid(),
    nn.Linear(3, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[0.5, 0.3], [-0.2, 0.4], [0.1, -0.6]]))
    model[0].bias.copy_(torch.tensor([0.1, -0.1, 0.2]))
    model[2].weight.copy_(torch.tensor([[0.4, -0.3, 0.2], [0.1, 0.5, -0.4]]))
    model[2].bias.copy_(torch.tensor([0.0, 0.1]))
    model[4].weight.copy_(torch.tensor([[0.6, -0.5]]))
    model[4].bias.copy_(torch.tensor([0.2]))
```

**(b)**

```python
x = torch.tensor([[1.0, 0.0]])
y = model(x)
print(y)  # Should output ≈ 0.5667
```

**(c)** Access the first layer's weight matrix:

```python
print(model[0].weight)      # shape: (3, 2)
print(model[0].weight.data)  # raw tensor data
```

---

## Part D: Loss and Backpropagation

### Problem 10: MSE Loss Computation

**(a)**
- $\ell_1 = \frac{1}{2}[(1.0-0.8)^2 + (0.0-0.2)^2] = \frac{1}{2}[0.04 + 0.04] = 0.04$
- $\ell_2 = \frac{1}{2}[(0.0-0.1)^2 + (1.0-0.7)^2] = \frac{1}{2}[0.01 + 0.09] = 0.05$
- $\ell_3 = \frac{1}{2}[(1.0-0.6)^2 + (1.0-0.9)^2] = \frac{1}{2}[0.16 + 0.01] = 0.085$

**(b)** $L = 0.04 + 0.05 + 0.085 = 0.175$

**(c)** Example 3 contributes the most ($\ell_3 = 0.085$). The network's prediction of $(0.6, 0.9)$ is furthest from target $(1.0, 1.0)$, particularly in the first output dimension ($0.6$ vs $1.0$). The network needs to improve its prediction of the first output for this example.

---

### Problem 11: Backpropagation on a Simple Network

**(a)** Forward pass:
- $z^{(0)} = 0.5(1.0) + 0.1 = 0.6$
- $a^{(0)} = \sigma(0.6) \approx 0.6457$
- $z^{(1)} = 0.8(0.6457) + (-0.2) = 0.5166 - 0.2 = 0.3166$
- $y = a^{(1)} = \sigma(0.3166) \approx 0.5785$

**(b)** $\ell = \frac{1}{2}(1.0 - 0.5785)^2 = \frac{1}{2}(0.4215)^2 = \frac{1}{2}(0.1777) \approx 0.0888$

**(c)** $\delta^{(1)} = -(\bar{y} - y) \cdot \sigma'(z^{(1)}) = -(1.0 - 0.5785) \cdot 0.5785(1 - 0.5785)$

$= -0.4215 \times 0.2439 = -0.1028$

**(d)**

Layer 1 gradients:
- $\frac{\partial \ell}{\partial w^{(1)}} = \delta^{(1)} \cdot a^{(0)} = -0.1028 \times 0.6457 = -0.0664$
- $\frac{\partial \ell}{\partial b^{(1)}} = \delta^{(1)} = -0.1028$

Backpropagate to layer 0:
$\delta^{(0)} = \delta^{(1)} \cdot w^{(1)} \cdot \sigma'(z^{(0)}) = -0.1028 \times 0.8 \times 0.6457(1 - 0.6457)$
$= -0.0823 \times 0.2289 = -0.01884$

Layer 0 gradients:
- $\frac{\partial \ell}{\partial w^{(0)}} = \delta^{(0)} \cdot x = -0.01884 \times 1.0 = -0.01884$
- $\frac{\partial \ell}{\partial b^{(0)}} = \delta^{(0)} = -0.01884$

**(e)** Updated parameters with $r = 0.5$:
- $w^{(1)}_{\text{new}} = 0.8 - 0.5(-0.0664) = 0.8 + 0.0332 = 0.8332$
- $b^{(1)}_{\text{new}} = -0.2 - 0.5(-0.1028) = -0.2 + 0.0514 = -0.1486$
- $w^{(0)}_{\text{new}} = 0.5 - 0.5(-0.01884) = 0.5 + 0.00942 = 0.5094$
- $b^{(0)}_{\text{new}} = 0.1 - 0.5(-0.01884) = 0.1 + 0.00942 = 0.1094$

All updates increase the weights/biases, which will push the output higher toward the target $\bar{y} = 1$.

---

### Problem 12: Backpropagation on a General Network

**(a)** Forward pass:

$$\vec{z}^{(0)} = \begin{pmatrix} 0.5 \\ -0.3 \end{pmatrix}(1.0) + \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix} = \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix}$$

$$\vec{a}^{(0)} = \begin{pmatrix} \sigma(0.6) \\ \sigma(-0.4) \end{pmatrix} \approx \begin{pmatrix} 0.6457 \\ 0.4013 \end{pmatrix}$$

$$z^{(1)} = 0.4(0.6457) + 0.6(0.4013) + 0.2 = 0.2583 + 0.2408 + 0.2 = 0.6991$$

$$y = \sigma(0.6991) \approx 0.6681$$

**(b)** With target $\bar{y} = 0$:

$$\delta^{(1)} = -(0 - 0.6681) \cdot \sigma'(0.6991) = 0.6681 \times 0.6681 \times (1 - 0.6681) = 0.6681 \times 0.2218 = 0.1482$$

Gradients for layer 1:

$$\frac{\partial \ell}{\partial \vec{w}^{(1)}} = \delta^{(1)} \cdot (\vec{a}^{(0)})^T = 0.1482 \times \begin{pmatrix} 0.6457 & 0.4013 \end{pmatrix} = \begin{pmatrix} 0.0957 & 0.0595 \end{pmatrix}$$

$$\frac{\partial \ell}{\partial b^{(1)}} = 0.1482$$

**(c)** Backpropagate to layer 0:

$$\vec{\delta}^{(0)} = \begin{pmatrix} 0.4 \\ 0.6 \end{pmatrix}(0.1482) \odot \begin{pmatrix} \sigma'(0.6) \\ \sigma'(-0.4) \end{pmatrix}$$

$$= \begin{pmatrix} 0.0593 \\ 0.0889 \end{pmatrix} \odot \begin{pmatrix} 0.6457 \times 0.3543 \\ 0.4013 \times 0.5987 \end{pmatrix} = \begin{pmatrix} 0.0593 \\ 0.0889 \end{pmatrix} \odot \begin{pmatrix} 0.2288 \\ 0.2403 \end{pmatrix}$$

$$= \begin{pmatrix} 0.01357 \\ 0.02136 \end{pmatrix}$$

Gradients for layer 0:

$$\frac{\partial \ell}{\partial W^{(0)}} = \vec{\delta}^{(0)} \cdot x = \begin{pmatrix} 0.01357 \\ 0.02136 \end{pmatrix}$$

$$\frac{\partial \ell}{\partial \vec{b}^{(0)}} = \begin{pmatrix} 0.01357 \\ 0.02136 \end{pmatrix}$$

**(d)** Update with $r = 1.0$:

- $\vec{w}^{(1)}_{\text{new}} = \begin{pmatrix} 0.4 - 0.0957 & 0.6 - 0.0595 \end{pmatrix} = \begin{pmatrix} 0.3043 & 0.5405 \end{pmatrix}$
- $b^{(1)}_{\text{new}} = 0.2 - 0.1482 = 0.0518$
- $W^{(0)}_{\text{new}} = \begin{pmatrix} 0.5 - 0.01357 \\ -0.3 - 0.02136 \end{pmatrix} = \begin{pmatrix} 0.4864 \\ -0.3214 \end{pmatrix}$
- $\vec{b}^{(0)}_{\text{new}} = \begin{pmatrix} 0.1 - 0.01357 \\ -0.1 - 0.02136 \end{pmatrix} = \begin{pmatrix} 0.0864 \\ -0.1214 \end{pmatrix}$

**(e)** Second forward pass with updated parameters:

$$\vec{z}^{(0)} = \begin{pmatrix} 0.4864 \\ -0.3214 \end{pmatrix}(1.0) + \begin{pmatrix} 0.0864 \\ -0.1214 \end{pmatrix} = \begin{pmatrix} 0.5728 \\ -0.4428 \end{pmatrix}$$

$$\vec{a}^{(0)} = \begin{pmatrix} \sigma(0.5728) \\ \sigma(-0.4428) \end{pmatrix} \approx \begin{pmatrix} 0.6394 \\ 0.3911 \end{pmatrix}$$

$$z^{(1)} = 0.3043(0.6394) + 0.5405(0.3911) + 0.0518 = 0.1946 + 0.2114 + 0.0518 = 0.4578$$

$$y_{\text{new}} = \sigma(0.4578) \approx 0.6124$$

**Yes**, the new prediction $0.6124$ is closer to the target $0.0$ than the old prediction $0.6681$. The loss decreased from $\frac{1}{2}(0.6681)^2 = 0.223$ to $\frac{1}{2}(0.6124)^2 = 0.188$.

---

## Part E: Conceptual and PyTorch

### Problem 13: True or False

**(a)** **False.** A single perceptron can only learn linearly separable functions. XOR is a Boolean function of two inputs that is not linearly separable, so a single perceptron cannot learn it.

**(b)** **False.** The sigmoid function outputs values in the **open** interval $(0, 1)$. It approaches but never reaches 0 or 1. For any finite input $x$, $0 < \sigma(x) < 1$.

**(c)** **True.** The backpropagation formula requires $f'(\vec{z}^{(l)})$ at every layer $l$. The derivative $f'$ is evaluated at the pre-activation values $\vec{z}^{(l)}$ computed during the forward pass, so these must be stored.

**(d)** **True.** With a very large learning rate, the weight update can overshoot the minimum on the loss surface. Instead of descending into the valley, the parameters jump across it to a point with higher loss. In extreme cases, the oscillations grow and the loss diverges.

**(e)** **False** in general, but **typically true in practice** with sigmoid. The vanishing gradient phenomenon causes gradients to shrink as they propagate backward, so earlier layers tend to have smaller gradients. However, this is not guaranteed — if weights are large, gradients could potentially grow (exploding gradients).

**(f)** **False.** PyTorch accumulates gradients by default. Without `zero_grad()`, each `loss.backward()` call adds new gradients to the existing ones, resulting in incorrect (accumulated) gradients. This is a logical error, not just an efficiency issue.

---

### Problem 14: Connecting the Chapters

**(a)** The Heaviside function has derivative $0$ everywhere except at $x = 0$ where it is undefined. Gradient descent requires $\frac{\partial L}{\partial w} = \delta \cdot a$, where $\delta$ involves $f'(z)$. If $f' = 0$, then $\delta = 0$ for all layers, making all gradients zero. The weights never update and the network cannot learn.

**(b)** Forward propagation must run first because backpropagation requires:
1. The **pre-activation values** $\vec{z}^{(l)}$ to compute $f'(\vec{z}^{(l)})$
2. The **activation values** $\vec{a}^{(l)}$ to compute weight gradients $\frac{\partial L}{\partial W^{(l)}} = \vec{\delta}^{(l)}(\vec{a}^{(l-1)})^T$
3. The **network output** $\vec{y}$ to compute the initial delta $\vec{\delta}^{(L)}$ from the loss

**(c)** The backpropagation recursion is exactly the **chain rule** applied layer by layer. For a quantity $z^{(l)}$ deep in the network:

$$\frac{\partial L}{\partial z^{(l)}} = \frac{\partial L}{\partial z^{(l+1)}} \cdot \frac{\partial z^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}}$$

In vector form:
- $\frac{\partial L}{\partial z^{(l+1)}} = \vec{\delta}^{(l+1)}$ (delta from next layer)
- $\frac{\partial z^{(l+1)}}{\partial a^{(l)}} = (W^{(l+1)})^T$ (the transpose redistributes error)
- $\frac{\partial a^{(l)}}{\partial z^{(l)}} = f'(\vec{z}^{(l)})$ (local derivative, applied elementwise)

Combining: $\vec{\delta}^{(l)} = [(W^{(l+1)})^T \vec{\delta}^{(l+1)}] \odot f'(\vec{z}^{(l)})$, which is the backpropagation recursion.

**(d)** With Heaviside activations, $f'(z) = 0$ for all $z \neq 0$. Therefore $\delta^{(l)} = [(W^{(l+1)})^T \delta^{(l+1)}] \odot f'(z^{(l)}) = \vec{0}$ for all layers (except in the measure-zero event that $z = 0$ exactly). With all deltas being zero, all weight gradients are zero, and no learning occurs.

---

### Problem 15: PyTorch Training

**(a)** and **(b):**

```python
import torch
import torch.nn as nn

# Define 2 → 4 → 4 → 1 network with sigmoid activations
model_sigmoid = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# XOR data
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model_sigmoid.parameters(), lr=1.0)

for epoch in range(10001):
    y_pred = model_sigmoid(X)
    loss = loss_fn(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

# Final predictions
with torch.no_grad():
    predictions = model_sigmoid(X)
    print("\nSigmoid Predictions:")
    for i in range(4):
        print(f"  {X[i].tolist()} → {predictions[i].item():.4f} (target: {Y[i].item():.0f})")
```

The network should converge to outputs close to the XOR targets (≈ 0 for (0,0) and (1,1), ≈ 1 for (0,1) and (1,0)), though convergence may take many epochs with sigmoid.

**(c)**

```python
model_tanh = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Sigmoid()   # keep output layer sigmoid for [0,1] output
)

optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), lr=1.0)

for epoch in range(10001):
    y_pred = model_tanh(X)
    loss = loss_fn(y_pred, Y)
    optimizer_tanh.zero_grad()
    loss.backward()
    optimizer_tanh.step()
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")
```

Tanh typically converges **faster** than sigmoid for this problem. At epoch 2,000, the tanh model usually has lower loss than the sigmoid model (e.g., tanh loss ≈ 0.001 vs sigmoid loss ≈ 0.05, though exact values depend on random initialization).

**(d)** Tanh has a maximum derivative of $1.0$ at $x = 0$, compared to sigmoid's maximum of $0.25$. This means:
- **Gradients are $4\times$ stronger** with tanh near $x = 0$
- Weight updates are proportionally larger per epoch
- The network makes more progress per training step

However, a learning rate that works well for sigmoid ($r = 1.0$) might be **too large** for tanh because the stronger gradients can cause overshooting. With tanh, a smaller learning rate (e.g., $r = 0.1$) may be needed for stable convergence. The product $r \cdot |\nabla L|$ determines the actual step size — with tanh's larger gradients, the effective step is larger even for the same $r$.
