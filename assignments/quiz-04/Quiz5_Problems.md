# IME 775 — Quiz 4   


### Question 1 (2 points)

Compute the following:

**(a)** (1 pt) $\sigma(0)=$ __________ and $\sigma'(0) =$ __________

**(b)** (1 pt) For the parametrized sigmoid with $w = 3$ and $b = 0$, compute the output at $x = 1$:

$$\sigma(1;\ 3,\ 0) = \frac{1}{1 + e^{-3}} \approx$$



### Question 2 (2 points)

Trace the forward pass through the following 2 → 2 → 1 MLP with sigmoid activations.

$$W^{(0)} = \begin{pmatrix} 1 & -1 \\ 0.5 & 0.5 \end{pmatrix}, \quad \vec{b}^{(0)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad \vec{w}^{(1)} = \begin{pmatrix} 1 & -1 \end{pmatrix}, \quad b^{(1)} = 0$$

Input: $\vec{x} = [1, 1]^T$.

**(a)** (1 pt) Compute $\vec{z}^{(0)}$ and $\vec{a}^{(0)}$.

**(b)** (1 pt) Compute $z^{(1)}$ and $y = a^{(1)}$.



### Question 3 (2 points)

A network produces output $y = 0.7$ for target $\bar{y} = 1.0$.

**(a)** (1 pt) Compute the MSE loss: $\ell = \frac{1}{2}(\bar{y} - y)^2 =$

**(b)** (1 pt) The pre-activation of the output neuron was $z = 0.847$. Compute the output-layer delta:

$$\delta = -(\bar{y} - y) \cdot \sigma'(z) = -(\bar{y} - y) \cdot y(1-y) =$$



### Question 4 (2 points)

In a 1 → 1 → 1 network (one neuron per layer, sigmoid activation), the forward pass gives:
- $a^{(0)} = 0.622$, $\delta^{(1)} = -0.05$, $w^{(1)} = 0.7$, $\sigma'(z^{(0)}) = 0.235$

**(a)** (1 pt) Compute the gradient of the loss with respect to $w^{(1)}$:

$$\frac{\partial \ell}{\partial w^{(1)}} = \delta^{(1)} \cdot a^{(0)} =$$

**(b)** (1 pt) Backpropagate: compute $\delta^{(0)}$ and the gradient $\frac{\partial \ell}{\partial w^{(0)}}$ given $x = 1.0$:

$$\delta^{(0)} = \delta^{(1)} \cdot w^{(1)} \cdot \sigma'(z^{(0)}) =$$

$$\frac{\partial \ell}{\partial w^{(0)}} = \delta^{(0)} \cdot x =$$



### Question 5 (2 points)

True or False (0.5 points each). Circle your answer.

**(a)** T / F — A single perceptron with Heaviside activation can compute the XOR function.

**(b)** T / F — The maximum value of $\sigma'(x)$ is $0.25$, which occurs at $x = 0$.

**(c)** T / F — In backpropagation, gradients are propagated from the input layer toward the output layer.

**(d)** T / F — In PyTorch, calling `loss.backward()` computes gradients for all parameters via backpropagation.
