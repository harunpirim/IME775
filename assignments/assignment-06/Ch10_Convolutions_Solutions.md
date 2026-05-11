# IME 775 – Assignment 3 Solutions (Chapter 10)
## Convolutions in Neural Networks

---

## Part A: 1D Convolution Fundamentals

### Problem 1

**(a)** $n = 8$, $k = 3$, $s = 1$, $p = 0$:

$$o = \left\lfloor \frac{8 - 3}{1} \right\rfloor + 1 = 5 + 1 = \boxed{6}$$

**(b)** Kernel $\vec{w} = [1, 0, -1]$ applied to $\vec{x} = [3, 1, 4, 1, 5, 9, 2, 6]$:

$$Y_0 = 1(3) + 0(1) + (-1)(4) = 3 - 4 = -1$$
$$Y_1 = 1(1) + 0(4) + (-1)(1) = 1 - 1 = 0$$
$$Y_2 = 1(4) + 0(1) + (-1)(5) = 4 - 5 = -1$$
$$Y_3 = 1(1) + 0(5) + (-1)(9) = 1 - 9 = -8$$
$$Y_4 = 1(5) + 0(9) + (-1)(2) = 5 - 2 = 3$$
$$Y_5 = 1(9) + 0(2) + (-1)(6) = 9 - 6 = 3$$

Output: $\vec{y} = [-1, 0, -1, -8, 3, 3]$

**(c)** The kernel $[1, 0, -1]$ computes the **difference** between the element two positions apart: $Y_x = X_x - X_{x+2}$. It detects edges (sharp changes) in the input — it is a discrete approximation of the first derivative.

**(d)** The largest absolute value is $|Y_3| = 8$ at index 3. This corresponds to the input region $[1, 5, 9]$ where the signal jumps sharply from 1 to 9 — a strong edge.

---

### Problem 2

**(a)** Using $o = \lfloor(n + 2p - k)/s\rfloor + 1$:

With zero padding $p=2$, the effective input size becomes $n + 2p = 10 + 4 = 14$:

| | Valid ($p=0$) | Zero padding ($p=2$) |
|---|---|---|
| $s=1$ | $\lfloor(10-4)/1\rfloor + 1 = \mathbf{7}$ | $\lfloor(10+4-4)/1\rfloor + 1 = \mathbf{11}$ |
| $s=2$ | $\lfloor(10-4)/2\rfloor + 1 = \mathbf{4}$ | $\lfloor(10+4-4)/2\rfloor + 1 = \mathbf{6}$ |
| $s=3$ | $\lfloor(10-4)/3\rfloor + 1 = \mathbf{3}$ | $\lfloor(10+4-4)/3\rfloor + 1 = \mathbf{4}$ |

**(b)** Stride $s=2$, valid padding. Kernel $= [\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}]$:

$$Y_0 = \tfrac{1}{4}(2 + 5 + 3 + 8) = \tfrac{18}{4} = 4.5$$
$$Y_1 = \tfrac{1}{4}(3 + 8 + 1 + 7) = \tfrac{19}{4} = 4.75$$
$$Y_2 = \tfrac{1}{4}(1 + 7 + 4 + 6) = \tfrac{18}{4} = 4.5$$

**(c)** This kernel computes the **local average** of 4 consecutive elements — a smoothing (low-pass) filter. In a noisy signal, the noise is high-frequency (rapid fluctuations). Averaging over neighbors smooths out the noise while preserving the broader trend.

---

### Problem 3

**(a)** Input size 4, kernel size 2, stride 1 → output size = $4 - 2 + 1 = 3$.

$$W = \begin{bmatrix} w_0 & w_1 & 0 & 0 \\ 0 & w_0 & w_1 & 0 \\ 0 & 0 & w_0 & w_1 \end{bmatrix}$$

**(b)** $W$ has dimensions $3 \times 4$ (output size $\times$ input size).

**(c)** Total entries: $3 \times 4 = 12$. Nonzero entries: $3 \times 2 = 6$. Zero entries: 6. Fraction of zeros = $6/12 = \boxed{50\%}$.

For larger inputs the fraction of zeros grows rapidly. With $n = 100$, $k = 2$: $W$ is $99 \times 100$. Nonzero = $99 \times 2 = 198$, total = $9900$. Zeros = $98\%$.

**(d)** With stride 2, the output size is $\lfloor(4-2)/2\rfloor + 1 = 2$. The kernel shifts by 2 positions per row:

$$W = \begin{bmatrix} w_0 & w_1 & 0 & 0 \\ 0 & 0 & w_0 & w_1 \end{bmatrix}$$

---

## Part B: 2D Convolution

### Problem 4

**(a)** $H=4$, $W=4$, $k_H=k_W=2$, stride $[1,1]$, valid:

$$o_H = 4 - 2 + 1 = 3, \qquad o_W = 4 - 2 + 1 = 3$$

Output size: $\boxed{3 \times 3}$.

**(b)** Kernel $W = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$. For each position $(r,c)$:

$$Y_{r,c} = X_{r,c} - X_{r,c+1} - X_{r+1,c} + X_{r+1,c+1}$$

$$Y_{0,0} = 10 - 20 - 50 + 60 = 0$$
$$Y_{0,1} = 20 - 30 - 60 + 70 = 0$$
$$Y_{0,2} = 30 - 40 - 70 + 80 = 0$$
$$Y_{1,0} = 50 - 60 - 90 + 100 = 0$$
$$Y_{1,1} = 60 - 70 - 100 + 110 = 0$$
$$Y_{1,2} = 70 - 80 - 110 + 120 = 0$$
$$Y_{2,0} = 90 - 100 - 130 + 140 = 0$$
$$Y_{2,1} = 100 - 110 - 140 + 150 = 0$$
$$Y_{2,2} = 110 - 120 - 150 + 160 = 0$$

$$Y = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

**(c)** This kernel detects **diagonal edges** (corners). It computes $X_{r,c} + X_{r+1,c+1} - X_{r,c+1} - X_{r+1,c}$. For a linearly varying image (constant gradient), the positive and negative parts cancel exactly, giving zero everywhere. The output would be nonzero only where the gradient changes direction (at corners or non-linear boundaries).

---

### Problem 5

**(a)** Smoothing kernel $W_s = \frac{1}{9}\mathbf{1}_{3\times3}$, stride $[1,1]$, valid. Output at $(r,c)$ = average of the $3 \times 3$ block starting at $(r,c)$:

$(0,0)$: avg of rows 0–2, cols 0–2: $\frac{100+100+100+100+100+100+10+10+10}{9} = \frac{630}{9} = \boxed{70.0}$

$(1,1)$: avg of rows 1–3, cols 1–2: $\frac{100+100+100+100+100+100+10+10+100}{9} = \frac{720}{9} = \boxed{80.0}$

$(2,2)$: avg of rows 2–4, cols 2–4: $\frac{10+100+100+10+100+100+10+10+10}{9} = \frac{450}{9} = \boxed{50.0}$

**(b)** Vertical edge kernel $W_e = \begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}$, stride $[1,1]$, valid:

$(1,1)$: $-X_{1,1} + X_{1,2} - X_{2,1} + X_{2,2} = -100 + 100 - 10 + 10 = \boxed{0}$

$(1,2)$: $-X_{1,2} + X_{1,3} - X_{2,2} + X_{2,3} = -100 + 100 - 10 + 100 = \boxed{90}$

$(2,1)$: $-X_{2,1} + X_{2,2} - X_{3,1} + X_{3,2} = -10 + 10 - 10 + 10 = \boxed{0}$

**(c)** The edge detection kernel produces a strong response of 90 at position $(1,2)$, which is exactly where the boundary between the dark region (10) and bright region (100) lies along the horizontal axis. The smoothing kernel blurs this transition. Edge detection highlights boundaries; smoothing suppresses them.

---

### Problem 6

**(a)** $o_H = \lfloor(224 + 2(1) - 3)/1\rfloor + 1 = 224$, same for $o_W$. Output: $\boxed{224 \times 224}$.

**(b)** $o_H = \lfloor(224 + 0 - 5)/2\rfloor + 1 = \lfloor 219/2\rfloor + 1 = 109 + 1 = 110$, same for $o_W$. Output: $\boxed{110 \times 110}$.

**(c)** $o_H = \lfloor(110 + 2(1) - 3)/2\rfloor + 1 = \lfloor 109/2\rfloor + 1 = 54 + 1 = 55$. Output: $\boxed{55 \times 55}$.

**(d)** Parameters per Conv2d layer = $C_{\text{out}} \times C_{\text{in}} \times k_H \times k_W$:

- Layer 1: $64 \times 3 \times 3 \times 3 = 1{,}728$
- Layer 2: $64 \times 64 \times 5 \times 5 = 102{,}400$
- Layer 3: $64 \times 64 \times 3 \times 3 = 36{,}864$

Total: $1{,}728 + 102{,}400 + 36{,}864 = \boxed{140{,}992}$

---

## Part C: 3D Convolution and Transposed Convolution

### Problem 7

**(a)**

$$o_T = 8 - 2 + 1 = 7, \quad o_H = 16 - 3 + 1 = 14, \quad o_W = 16 - 3 + 1 = 14$$

Output size: $\boxed{7 \times 14 \times 14}$.

**(b)** No motion — both frames have identical patch values:

$$Y = (-1)(50)(9) + (1)(50)(9) = -450 + 450 = \boxed{0}$$

The zero output correctly indicates no motion at this location.

**(c)** Motion present:

$$Y = (-1)(50)(9) + (1)(200)(9) = -450 + 1800 = \boxed{1350}$$

The large positive value indicates strong motion — pixel values changed significantly between the two frames at this spatial location.

---

### Problem 8

**(a)** Input size 5, kernel size 3, stride 1, valid → output size = $5 - 3 + 1 = 3$.

$$W = \begin{bmatrix} w_0 & w_1 & w_2 & 0 & 0 \\ 0 & w_0 & w_1 & w_2 & 0 \\ 0 & 0 & w_0 & w_1 & w_2 \end{bmatrix}$$

**(b)**

$$W^T = \begin{bmatrix} w_0 & 0 & 0 \\ w_1 & w_0 & 0 \\ w_2 & w_1 & w_0 \\ 0 & w_2 & w_1 \\ 0 & 0 & w_2 \end{bmatrix}$$

**(c)** With $\vec{y} = [2, -1, 3]$:

$$\tilde{x} = W^T\vec{y} = \begin{bmatrix} 2w_0 \\ 2w_1 - w_0 \\ 2w_2 - w_1 + 3w_0 \\ -w_2 + 3w_1 \\ 3w_2 \end{bmatrix}$$

**(d)** No, $\tilde{x} \neq \vec{x}$. The forward convolution maps 5 independent values to 3 values — information is irretrievably lost. The weight matrix $W$ is $3 \times 5$ (non-square, non-invertible), so there is no $W^{-1}$. The transposed operation distributes the output back in the same proportions as the forward collected it, but it cannot recover the original input.

**(e)** $o' = (n' - 1) \times s + k - 2p = (3-1)(2) + 3 - 0 = 4 + 3 = \boxed{7}$

---

## Part D: Pooling

### Problem 9

**(a)** $2 \times 2$ max pooling, stride 2 on the $6 \times 6$ input:

$$\text{MaxPool} = \begin{bmatrix} \max(1,3,7,4) & \max(2,8,6,1) & \max(0,5,3,2) \\ \max(0,9,3,2) & \max(5,4,8,6) & \max(7,1,0,9) \\ \max(5,1,6,7) & \max(0,3,2,1) & \max(8,4,5,3) \end{bmatrix} = \begin{bmatrix} 7 & 8 & 5 \\ 9 & 8 & 9 \\ 7 & 3 & 8 \end{bmatrix}$$

**(b)** $2 \times 2$ average pooling, stride 2:

$$\text{AvgPool} = \begin{bmatrix} \frac{1+3+7+4}{4} & \frac{2+8+6+1}{4} & \frac{0+5+3+2}{4} \\ \frac{0+9+3+2}{4} & \frac{5+4+8+6}{4} & \frac{7+1+0+9}{4} \\ \frac{5+1+6+7}{4} & \frac{0+3+2+1}{4} & \frac{8+4+5+3}{4} \end{bmatrix} = \begin{bmatrix} 3.75 & 4.25 & 2.50 \\ 3.50 & 5.75 & 4.25 \\ 4.75 & 1.50 & 5.00 \end{bmatrix}$$

**(c)** $3 \times 3$ max pooling, stride 3 → output size: $\lfloor(6-3)/3\rfloor + 1 = 2$ per dimension → $2 \times 2$.

$$\text{MaxPool}_{3\times3} = \begin{bmatrix} \max(\text{rows 0-2, cols 0-2}) & \max(\text{rows 0-2, cols 3-5}) \\ \max(\text{rows 3-5, cols 0-2}) & \max(\text{rows 3-5, cols 3-5}) \end{bmatrix} = \begin{bmatrix} 9 & 8 \\ 8 & 9 \end{bmatrix}$$

**(d)** Max pooling retains the **strongest activation** in each neighborhood, which corresponds to the most confident feature detection. Average pooling dilutes strong activations with weaker ones. For classification, we care about *whether* a feature is present (max), not the average response. Max pooling also provides better gradient flow during backpropagation — the gradient passes only through the maximum element, creating a sparser but more informative signal.

---

### Problem 10

**(a)**

| Layer | Configuration | Output Size |
|---|---|---|
| Input | — | $1 \times 28 \times 28$ |
| Conv2d | 16 filters, $3 \times 3$, stride 1, pad 1 | $16 \times 28 \times 28$ |
| ReLU | — | $16 \times 28 \times 28$ |
| MaxPool2d | $2 \times 2$, stride 2 | $16 \times 14 \times 14$ |
| Conv2d | 32 filters, $3 \times 3$, stride 1, pad 1 | $32 \times 14 \times 14$ |
| ReLU | — | $32 \times 14 \times 14$ |
| MaxPool2d | $2 \times 2$, stride 2 | $32 \times 7 \times 7$ |
| Flatten | — | $1568$ |
| Linear | 10 outputs | $10$ |

**(b)** Conv2d parameters = $C_{\text{out}} \times (C_{\text{in}} \times k_H \times k_W + 1)$ (the +1 is for bias per filter):

- Layer 1: $16 \times (1 \times 3 \times 3 + 1) = 16 \times 10 = \mathbf{160}$
- Layer 2: $32 \times (16 \times 3 \times 3 + 1) = 32 \times 145 = \mathbf{4{,}640}$

**(c)** Linear layer: $1568 \times 10 + 10 = \mathbf{15{,}690}$ (weights + biases).

**(d)** Total CNN parameters: $160 + 4{,}640 + 15{,}690 = 20{,}490$.

A single FC layer from 784 to 10: $784 \times 10 + 10 = 7{,}850$.

Ratio: $20{,}490 / 7{,}850 \approx 2.6\times$.

The CNN has more total parameters, but its convolutional layers have very few parameters (4,800 combined) compared to the FC layer's 7,850. The bulk of the CNN's parameters come from the flatten-to-FC transition. Despite the higher count, the CNN will generalize far better because the convolutional layers exploit spatial locality and weight sharing.

---

*IME 775 – Mathematical Foundations of Deep Learning*
