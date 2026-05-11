# IME 775 — Quiz 2: Convolutions in Neural Networks

**Course:** IME 775 - Mathematical Foundations of Deep Learning
**Reference:** Chapter 10
**Time:** 25 minutes

---

### Question 1 (10 pts) — 1D Convolution

Given input $\vec{x} = [2, 5, 1, 3, 7]$ and kernel $\vec{w} = [1, -2, 1]$ with stride 1 and valid padding:

**(a)** (3 pts) How many output elements are produced?

**(b)** (7 pts) Compute all output values.

---

### Question 2 (10 pts) — Output Size

A 2D convolution has:
- Input size: $32 \times 32$
- Kernel size: $5 \times 5$
- Stride: $[2, 2]$
- Padding: $p = 2$

**(a)** (5 pts) What is the spatial output size? Show your work.

**(b)** (5 pts) $2 \times 2$ max pooling with stride 2 is applied to the result of part (a). What is the final output size?

---

### Question 3 (10 pts) — 2D Convolution

Compute the $2 \times 2$ output of convolving the input with the given kernel (stride 1, valid):

$$X = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \\ 0 & 2 & 4 \end{bmatrix}, \qquad W = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}$$

---

### Question 4 (10 pts) — Conceptual Questions

Answer in 2–3 sentences each.

**(a)** (5 pts) What two properties make convolutional layers more parameter-efficient than fully connected layers for image inputs?

**(b)** (5 pts) What is the purpose of transposed convolution? Give one application.

---

### Question 5 (10 pts) — Pooling

Given the $4 \times 4$ feature map:

$$F = \begin{bmatrix} 3 & 1 & 7 & 2 \\ 0 & 5 & 4 & 8 \\ 6 & 2 & 1 & 3 \\ 9 & 0 & 5 & 4 \end{bmatrix}$$

**(a)** (5 pts) Apply $2 \times 2$ max pooling with stride 2.

**(b)** (5 pts) Apply $2 \times 2$ average pooling with stride 2.

---

*Total: 50 points*
*IME 775 — Mathematical Foundations of Deep Learning*
