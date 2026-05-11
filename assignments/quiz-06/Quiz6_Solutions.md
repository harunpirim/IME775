# IME 775 — Quiz 2 Solutions: Convolutions in Neural Networks

---

### Question 1 (10 pts) — 1D Convolution

Input $\vec{x} = [2, 5, 1, 3, 7]$, kernel $\vec{w} = [1, -2, 1]$, stride 1, valid padding.

**(a)** (3 pts)

$$o = \left\lfloor\frac{n - k}{s}\right\rfloor + 1 = \left\lfloor\frac{5 - 3}{1}\right\rfloor + 1 = \boxed{3}$$

**(b)** (7 pts)

$$Y_0 = 1(2) + (-2)(5) + 1(1) = 2 - 10 + 1 = \boxed{-7}$$
$$Y_1 = 1(5) + (-2)(1) + 1(3) = 5 - 2 + 3 = \boxed{6}$$
$$Y_2 = 1(1) + (-2)(3) + 1(7) = 1 - 6 + 7 = \boxed{2}$$

Output: $\vec{y} = [-7, 6, 2]$

*Note:* The kernel $[1, -2, 1]$ is a discrete second-derivative operator (Laplacian). It detects changes in the rate of change — useful for detecting peaks and troughs.

---

### Question 2 (10 pts) — Output Size

**(a)** (5 pts)

$$o_H = \left\lfloor\frac{32 + 2(2) - 5}{2}\right\rfloor + 1 = \left\lfloor\frac{31}{2}\right\rfloor + 1 = 15 + 1 = \boxed{16}$$

Same for $o_W$. Output: $16 \times 16$.

**(b)** (5 pts)

$$o_H = \left\lfloor\frac{16 - 2}{2}\right\rfloor + 1 = 7 + 1 = \boxed{8}$$

After pooling: $8 \times 8$.

---

### Question 3 (10 pts) — 2D Convolution

$$X = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \\ 0 & 2 & 4 \end{bmatrix}, \quad W = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}$$

Output size: $(3-2+1) \times (3-2+1) = 2 \times 2$.

$$Y_{0,0} = 1(1) + (-1)(0) + 0(3) + 1(1) = 1 + 0 + 0 + 1 = \boxed{2}$$

$$Y_{0,1} = 1(0) + (-1)(2) + 0(1) + 1(0) = 0 - 2 + 0 + 0 = \boxed{-2}$$

$$Y_{1,0} = 1(3) + (-1)(1) + 0(0) + 1(2) = 3 - 1 + 0 + 2 = \boxed{4}$$

$$Y_{1,1} = 1(1) + (-1)(0) + 0(2) + 1(4) = 1 + 0 + 0 + 4 = \boxed{5}$$

$$Y = \begin{bmatrix} 2 & -2 \\ 4 & 5 \end{bmatrix}$$

---

### Question 4 (10 pts) — Conceptual Questions

**(a)** (5 pts)

**Weight sharing** — the same small kernel is applied at every spatial position, so the number of parameters depends only on the kernel size, not the input size.

**Local connectivity** — each output neuron connects to only a small spatial region of the input (the receptive field), rather than to all input neurons as in a fully connected layer.

Together, these reduce parameters from $O(n^2)$ (fully connected) to $O(k^2)$ (convolutional), where $k \ll n$.

**(b)** (5 pts)

Transposed convolution performs **learnable upsampling** — it maps a smaller spatial representation to a larger one while using learned weights. It is the natural counterpart (adjoint) of the forward convolution.

**Application:** In an **autoencoder** (or U-Net), the encoder compresses the input through convolution and pooling; the decoder uses transposed convolutions to reconstruct the original resolution.

---

### Question 5 (10 pts) — Pooling

$$F = \begin{bmatrix} 3 & 1 & 7 & 2 \\ 0 & 5 & 4 & 8 \\ 6 & 2 & 1 & 3 \\ 9 & 0 & 5 & 4 \end{bmatrix}$$

**(a)** (5 pts) $2 \times 2$ max pooling, stride 2:

$$\text{MaxPool} = \begin{bmatrix} \max(3,1,0,5) & \max(7,2,4,8) \\ \max(6,2,9,0) & \max(1,3,5,4) \end{bmatrix} = \boxed{\begin{bmatrix} 5 & 8 \\ 9 & 5 \end{bmatrix}}$$

**(b)** (5 pts) $2 \times 2$ average pooling, stride 2:

$$\text{AvgPool} = \begin{bmatrix} \frac{3+1+0+5}{4} & \frac{7+2+4+8}{4} \\ \frac{6+2+9+0}{4} & \frac{1+3+5+4}{4} \end{bmatrix} = \boxed{\begin{bmatrix} 2.25 & 5.25 \\ 4.25 & 3.25 \end{bmatrix}}$$

---

*Total: 50 points*
*IME 775 — Mathematical Foundations of Deep Learning*
