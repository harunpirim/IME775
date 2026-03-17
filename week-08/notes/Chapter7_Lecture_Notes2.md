# IME 775 — Lecture 14
## Multilayer Perceptrons, Cybenko's Theorem, and Universal Approximation

---

## 1. The XOR Problem: Why We Need Multiple Layers

In Lecture 13, we saw that XOR cannot be modeled with a single perceptron because the classes are not linearly separable. The solution: **connect multiple perceptrons into a network**.

### XOR via an MLP

The XOR function $y = x_0 \oplus x_1$ can be decomposed as:

$$y = \bar{x}_0 x_1 + x_0 \bar{x}_1$$

This is a sum (OR) of two AND terms, each involving one negated input:
- Term 1: $\bar{x}_0 x_1$ — NOT($x_0$) AND $x_1$
- Term 2: $x_0 \bar{x}_1$ — $x_0$ AND NOT($x_1$)

Each term can be implemented by a single perceptron, and their OR can be implemented by another perceptron. This gives us a **2-layer network**.

### Layer 0 (Hidden Layer)

**Perceptron for $\bar{x}_0 x_1$:**

$$h_0 = \theta(-x_0 + x_1 - 0.5)$$

| $x_0$ | $x_1$ | $z = -x_0 + x_1 - 0.5$ | $h_0$ |
|:---:|:---:|:---:|:---:|
| 0 | 0 | $-0.5$ | 0 |
| 0 | 1 | $0.5$ | 1 |
| 1 | 0 | $-1.5$ | 0 |
| 1 | 1 | $-0.5$ | 0 |

**Perceptron for $x_0 \bar{x}_1$:**

$$h_1 = \theta(x_0 - x_1 - 0.5)$$

| $x_0$ | $x_1$ | $z = x_0 - x_1 - 0.5$ | $h_1$ |
|:---:|:---:|:---:|:---:|
| 0 | 0 | $-0.5$ | 0 |
| 0 | 1 | $-1.5$ | 0 |
| 1 | 0 | $0.5$ | 1 |
| 1 | 1 | $-0.5$ | 0 |

### Layer 1 (Output Layer)

**OR of hidden outputs:**

$$y = \theta(h_0 + h_1 - 0.5)$$

### Complete XOR Computation

| $x_0$ | $x_1$ | $h_0$ | $h_1$ | $y = \theta(h_0 + h_1 - 0.5)$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 | $\theta(-0.5) = 0$ |
| 0 | 1 | 1 | 0 | $\theta(0.5) = 1$ |
| 1 | 0 | 0 | 1 | $\theta(0.5) = 1$ |
| 1 | 1 | 0 | 0 | $\theta(-0.5) = 0$ |

This matches the XOR truth table exactly.

### Matrix Form

The XOR MLP can be written compactly:

**Layer 0:** $\vec{h} = \theta\left(W^{(0)}\vec{x} + \vec{b}^{(0)}\right)$ where $W^{(0)} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}$, $\vec{b}^{(0)} = \begin{pmatrix} -0.5 \\ -0.5 \end{pmatrix}$

**Layer 1:** $y = \theta\left(\vec{w}^{(1)T}\vec{h} + b^{(1)}\right)$ where $\vec{w}^{(1)} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $b^{(1)} = -0.5$

&nbsp;

*Workout:* Verify the XOR MLP for input $\vec{x} = (1, 0)^T$ by computing each layer step by step.

**Solution:**

Layer 0:
$$\vec{z}^{(0)} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \begin{pmatrix} -0.5 \\ -0.5 \end{pmatrix} = \begin{pmatrix} -1 \\ 1 \end{pmatrix} + \begin{pmatrix} -0.5 \\ -0.5 \end{pmatrix} = \begin{pmatrix} -1.5 \\ 0.5 \end{pmatrix}$$

$$\vec{h} = \theta(\vec{z}^{(0)}) = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

Layer 1:
$$z^{(1)} = \begin{pmatrix} 1 & 1 \end{pmatrix}\begin{pmatrix} 0 \\ 1 \end{pmatrix} + (-0.5) = 1 - 0.5 = 0.5$$

$$y = \theta(0.5) = 1 \quad \checkmark$$

---

## 2. MLP Architecture: Layering

### Rules of MLP Organization

1. **Layers are numbered** with increasing integers from input to output
2. **Feed-forward only:** the output of a perceptron in layer $i$ is fed only to perceptrons in layer $i+1$ — no skip connections, no backward connections
3. **Hidden layers:** all layers except the last are invisible (their outputs are not directly accessible)
4. **Weight indexing:** each weight and bias belongs to one layer, indicated by superscript: $w^{(l)}$, $b^{(l)}$

### Architecture Diagram

```
Input       Hidden Layer(s)      Output
 x₀ ──┐   ┌── h₀ ──┐
 x₁ ──┼───┤         ├───── y
 x₂ ──┘   └── h₁ ──┘
       Layer 0      Layer 1
```

### Depth and the "Deep" in Deep Learning

| Network | Hidden layers | Terminology |
|---|---|---|
| Single perceptron | 0 | Linear classifier |
| 1 hidden layer | 1 | Shallow network |
| 2+ hidden layers | 2+ | **Deep neural network** |

**MLPs with two or more hidden layers are called deep neural networks — this is the origin of "deep learning."**

---

## 3. Modeling Arbitrary Logical Functions with MLPs

**Claim:** Any logical function (any truth table) can be implemented as an MLP.

### The Construction

Any truth table can be converted to an MLP mechanically:

**Step 1:** Identify all rows where the output is 1.

**Step 2:** For each such row, construct an AND perceptron that fires only for that specific input combination. Use the input directly if the value is 1, and the negated input (negative weight) if the value is 0.

**Step 3:** OR all the AND outputs together.

This is the **sum-of-products** (disjunctive normal form) — identical to the standard approach in digital logic design.

&nbsp;

*Workout:* Construct an MLP for the truth table:

| $x_0$ | $x_1$ | $y$ |
|:---:|:---:|:---:|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Solution:**
This is XNOR ($y = \overline{x_0 \oplus x_1}$). Rows with $y = 1$: $(0,0)$ and $(1,1)$.

- Term 1 ($\bar{x}_0 \bar{x}_1$): $h_0 = \theta(-x_0 - x_1 + 0.5)$ — fires only at $(0,0)$
- Term 2 ($x_0 x_1$): $h_1 = \theta(x_0 + x_1 - 1.5)$ — fires only at $(1,1)$
- Output (OR): $y = \theta(h_0 + h_1 - 0.5)$

Verification: $(0,0) \to h_0=1, h_1=0 \to y=1$. $(0,1) \to h_0=0, h_1=0 \to y=0$. $(1,0) \to h_0=0, h_1=0 \to y=0$. $(1,1) \to h_0=0, h_1=1 \to y=1$. ✓

---

## 4. Cybenko's Universal Approximation Theorem

### The Core Idea

**Theorem (Cybenko, 1989):** Any continuous function $f: [a, b] \to \mathbb{R}$ can be approximated to arbitrary accuracy by an MLP with a single hidden layer and a finite number of neurons.

### Intuition: Approximation by Towers

Any continuous function can be approximated by a sum of **towers** (rectangles):
- Thinner towers → more towers → better approximation
- This is a direct consequence of the **mean value theorem for integrals** from calculus

The key insight is that each tower can be built from perceptrons, so the entire function can be approximated by an MLP.

### Building a 1D Tower from Steps

**Step 1: The basic step**

$$\theta(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}$$

**Step 2: Shift the step** — a bias moves the step left or right:

$$\theta(x + c)$$ shifts the step to $x = -c$

**Step 3: Flip the step** — a negative weight mirrors it:

$$\theta(-x)$$ creates a step that goes from 1 to 0

**Step 4: Combine into a tower** — AND a regular step with a flipped-and-shifted step:

$$\text{tower}(x) = \theta\left(\theta(x + a) + \theta(-x + b) - 1.5\right)$$

This creates a rectangular pulse that is 1 between $x = -a$ and $x = b$, and 0 elsewhere.

&nbsp;

*Workout:* Construct a 1D tower MLP that outputs 1 for $x \in [-3, 3]$ and 0 otherwise. Specify all weights and biases.

**Solution:**

We need a left step at $x = -3$ and a flipped right step at $x = 3$:

**Hidden layer (2 perceptrons):**
- $h_0 = \theta(x + 3)$: weight $w_0^{(0)} = 1$, bias $b_0^{(0)} = 3$ (step at $x = -3$)
- $h_1 = \theta(-x + 3)$: weight $w_1^{(0)} = -1$, bias $b_1^{(0)} = 3$ (flipped step at $x = 3$)

**Output layer (AND):**
- $y = \theta(h_0 + h_1 - 1.5)$: weights $w^{(1)} = (1, 1)$, bias $b^{(1)} = -1.5$

Verification:
- $x = -5$: $h_0 = \theta(-2) = 0$, $h_1 = \theta(8) = 1$ → $y = \theta(-0.5) = 0$ ✓
- $x = 0$: $h_0 = \theta(3) = 1$, $h_1 = \theta(3) = 1$ → $y = \theta(0.5) = 1$ ✓
- $x = 5$: $h_0 = \theta(8) = 1$, $h_1 = \theta(-2) = 0$ → $y = \theta(-0.5) = 0$ ✓

### Multiple Towers → Function Approximation

To approximate a function $f(x)$ on $[a, b]$:

1. Divide $[a, b]$ into $n$ intervals
2. For each interval, create a tower with height $\approx f(\text{midpoint})$
3. Sum all towers (with appropriate height scaling)

More intervals (thinner towers) → better approximation.

**Nuanced point — Cybenko guarantees existence, not practicality:**

The theorem says *a single hidden layer suffices*, but the number of neurons in that layer can be **arbitrarily large**. For a complicated function with many wiggles, you might need millions of towers. This is why in practice we use **multiple hidden layers (deep networks)** instead of one enormous layer. Depth allows the network to build hierarchical features — each layer captures increasingly abstract patterns — leading to far fewer total parameters than a single wide layer would require.

**Deeper explanation — why depth beats width:**

Consider approximating a function with $n$ oscillations. A single hidden layer needs $O(n)$ neurons (one tower per oscillation). But a deep network can compose simpler functions: layer 1 creates basic features, layer 2 combines them, and so on. This compositional structure can represent the same function with $O(\log n)$ neurons per layer and $O(\log n)$ layers — exponentially fewer parameters. This is analogous to how a $k$-digit number can represent $10^k$ values: composition is exponentially more efficient than enumeration.

---

## 5. 2D Towers and Higher Dimensions

### 2D Steps

A 2D step along $x_0$:

$$y = \theta\left(\begin{pmatrix} 1 & 0 \end{pmatrix}\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}\right) = \theta(x_0)$$

This is a "wall" perpendicular to the $x_0$ axis.

### 2D Waves

Combining a step with its flipped version along one axis creates a **wave** — a band that is 1 in a strip and 0 elsewhere:

$$\text{wave}_{x_0}(x_0, x_1) = \theta\left(\theta(x_0 + a) + \theta(-x_0 + b) - 1.5\right)$$

### 2D Tower = AND of Two Waves

A 2D tower is the intersection of a wave along $x_0$ and a wave along $x_1$:

$$\text{tower}(x_0, x_1) = \text{wave}_{x_0} \wedge \text{wave}_{x_1}$$

This creates a rectangular column that is 1 inside a rectangle and 0 outside. The MLP has:
- **4 perceptrons** in hidden layer 0 (two steps per axis, two axes)
- **2 perceptrons** in hidden layer 1 (one wave per axis)
- **1 perceptron** in the output layer (AND of the two waves)

### Generalizing to $d$ Dimensions

In $d$ dimensions, a tower requires waves along each axis. The number of hidden-layer perceptrons scales linearly with dimension, not exponentially — this is architecturally manageable.

&nbsp;

*Workout:* A 2D tower is centered at the origin and spans $x_0 \in [-2, 2]$, $x_1 \in [-1, 1]$. Write the weight matrices and bias vectors for each layer of the MLP.

**Solution:**

**Layer 0 (4 perceptrons — one step per boundary):**

$$W^{(0)} = \begin{pmatrix} 1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{pmatrix}, \quad \vec{b}^{(0)} = \begin{pmatrix} 2 \\ 2 \\ 1 \\ 1 \end{pmatrix}$$

- $h_0 = \theta(x_0 + 2)$: left boundary at $x_0 = -2$
- $h_1 = \theta(-x_0 + 2)$: right boundary at $x_0 = 2$
- $h_2 = \theta(x_1 + 1)$: bottom boundary at $x_1 = -1$
- $h_3 = \theta(-x_1 + 1)$: top boundary at $x_1 = 1$

**Layer 1 (AND of all 4):**

$$\vec{w}^{(1)} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}, \quad b^{(1)} = -3.5$$

Output $y = \theta(h_0 + h_1 + h_2 + h_3 - 3.5)$ fires only when all four boundaries are satisfied.

---

## 6. MLPs for Polygonal Decision Boundaries

### Rectangular Decision Region

Consider classifying points inside the rectangle $x_0 \in [-5, 5]$, $x_1 \in [-2, 2]$.

Each edge of the rectangle defines a half-plane:
- $x_0 \geq -5$: perceptron $\theta(x_0 + 5)$
- $x_0 \leq 5$: perceptron $\theta(-x_0 + 5)$
- $x_1 \geq -2$: perceptron $\theta(x_1 + 2)$
- $x_1 \leq 2$: perceptron $\theta(-x_1 + 2)$

ANDing all four: the output fires only inside the rectangle.

### From Rectangles to Arbitrary Shapes

1. **Any polygon** can be represented as an intersection of half-planes → single MLP layer with one perceptron per edge
2. **Any shape** on a plane can be approximated by a polygon with enough edges
3. Therefore, MLPs can approximate **any decision boundary** to arbitrary accuracy

&nbsp;

*Workout:* Design an MLP for a triangular decision region with vertices at $(0, 0)$, $(4, 0)$, and $(2, 3)$.

**Solution:**

The three edges of the triangle define three half-planes. We need the inequalities that are satisfied inside the triangle:

1. Bottom edge ($y \geq 0$): $x_1 \geq 0$ → $\theta(x_1)$
2. Left edge (from $(0,0)$ to $(2,3)$): $3x_0 - 2x_1 \geq 0$... more precisely, the line is $x_1 = \frac{3}{2}x_0$, and the interior is below it: $-3x_0 + 2x_1 \leq 0$ → $\theta(3x_0 - 2x_1)$
3. Right edge (from $(4,0)$ to $(2,3)$): the line is $x_1 = -\frac{3}{2}(x_0 - 4) = -\frac{3}{2}x_0 + 6$, and the interior is below it: $3x_0 + 2x_1 - 12 \leq 0$ → $\theta(-3x_0 - 2x_1 + 12)$

**Hidden layer:**
- $h_0 = \theta(x_1)$: $\vec{w}_0 = (0, 1)$, $b_0 = 0$
- $h_1 = \theta(3x_0 - 2x_1)$: $\vec{w}_1 = (3, -2)$, $b_1 = 0$
- $h_2 = \theta(-3x_0 - 2x_1 + 12)$: $\vec{w}_2 = (-3, -2)$, $b_2 = 12$

**Output (AND of all 3):**
- $y = \theta(h_0 + h_1 + h_2 - 2.5)$

Verify centroid $(2, 1)$: $h_0 = \theta(1) = 1$, $h_1 = \theta(4) = 1$, $h_2 = \theta(-6-2+12) = \theta(4) = 1$ → $y = \theta(0.5) = 1$ ✓

Verify outside point $(0, 2)$: $h_0 = \theta(2) = 1$, $h_1 = \theta(-4) = 0$, $h_2 = \theta(8) = 1$ → $y = \theta(-0.5) = 0$ ✓

---

## 7. Expressive Power: Putting It All Together

### The Hierarchy of Representable Functions

| Architecture | What it can represent | Limitation |
|---|---|---|
| Single perceptron | Linear decision boundary (hyperplane) | Cannot do XOR |
| 1 hidden layer MLP | Any continuous function (Cybenko) | May need impractically many neurons |
| Deep MLP (2+ hidden layers) | Same, but with far fewer parameters | Harder to train (vanishing gradients) |

### Architecture vs. Parameters

| Aspect | Controls | Analogy |
|---|---|---|
| **Architecture** | What functions *could* be represented | The blueprint of a building |
| **Parameters** ($\vec{w}, b$) | Which specific function *is* represented | The materials and dimensions |

**Training** finds the parameter values that make the network approximate the target function as closely as possible. We discuss training in Chapter 8.

### The Evolution of Neural Network Thinking

```
1958: Perceptron (Rosenblatt)
 │    Linear classifier, single neuron
 │
1969: Minsky & Papert expose XOR limitation
 │    Single perceptrons can't solve non-linearly separable problems
 │    → "AI Winter" — funding and interest decline
 │
1986: Backpropagation (Rumelhart, Hinton, Williams)
 │    Efficient training of MLPs with multiple layers
 │    → Revival of neural networks
 │
1989: Cybenko's Universal Approximation Theorem
 │    Proves MLPs are universal function approximators
 │
2006+: Deep Learning revolution
 │    Many hidden layers + large data + GPU compute
 │    → State-of-the-art in vision, language, games
 │
2012: AlexNet wins ImageNet
 │    Deep CNNs dramatically outperform all other methods
 │
2017+: Transformers and foundation models
      Attention mechanisms, massive scale
```

---

## 8. Connection to Upcoming Topics

| This chapter | Next chapters |
|---|---|
| Step function activation | Sigmoid, ReLU, and other smooth activations (Ch. 8) |
| Manual weight assignment | Automated training via backpropagation (Ch. 8) |
| Logical functions | Real-world loss functions (Ch. 9) |
| Generic MLP architecture | Specialized architectures: CNNs, etc. (Ch. 10–11) |
| Cybenko (single hidden layer) | Deep networks with multiple layers (Ch. 8–11) |

---

## Key Takeaways

1. **XOR** is the simplest function that proves single perceptrons are insufficient → need for multilayer networks
2. **MLPs** organize perceptrons into layers; outputs of layer $i$ feed into layer $i+1$ only
3. Any **truth table** can be mechanically converted to an MLP via sum-of-products decomposition
4. **Cybenko's theorem** guarantees that a single hidden layer can approximate any continuous function — but may require impractically many neurons
5. **Depth beats width**: deep networks achieve the same expressive power with exponentially fewer parameters than wide shallow ones
6. **Tower construction** (1D and 2D) is the concrete proof mechanism for Cybenko's theorem: steps → waves → towers → function approximation
7. **Polygonal decision boundaries** can be modeled by ANDing half-plane perceptrons — and any shape can be approximated by polygons

---

## PyTorch Connection

```python
import torch

def heaviside(x):
    return (x >= 0).float()

def mlp_xor(X):
    """XOR via a 2-layer MLP."""
    W0 = torch.tensor([[-1.0, 1.0],
                        [ 1.0, -1.0]])
    b0 = torch.tensor([-0.5, -0.5])
    
    w1 = torch.tensor([1.0, 1.0])
    b1 = torch.tensor(-0.5)
    
    h = heaviside(X @ W0.T + b0)
    y = heaviside(h @ w1 + b1)
    return y

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
print("XOR outputs:", mlp_xor(X))  # [0, 1, 1, 0]

def tower_1d(x, left, right):
    """1D tower: returns 1 if left <= x <= right."""
    h0 = heaviside(x - left)
    h1 = heaviside(-x + right)
    return heaviside(h0 + h1 - 1.5)

x = torch.linspace(-5, 5, 100)
y = tower_1d(x, -3, 3)
print(f"Tower active for {(y == 1).sum()} of {len(x)} points")

def tower_2d(X, x0_min, x0_max, x1_min, x1_max):
    """2D tower: returns 1 inside the rectangle."""
    h0 = heaviside(X[:, 0] - x0_min)
    h1 = heaviside(-X[:, 0] + x0_max)
    h2 = heaviside(X[:, 1] - x1_min)
    h3 = heaviside(-X[:, 1] + x1_max)
    return heaviside(h0 + h1 + h2 + h3 - 3.5)

pts = torch.tensor([[0, 0], [3, 1], [-6, 0], [2, -3]], dtype=torch.float)
print("Inside [-5,5]x[-2,2]:", tower_2d(pts, -5, 5, -2, 2))
```
