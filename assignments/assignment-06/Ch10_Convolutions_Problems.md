# Assignment 6  — Convolutions in Neural Networks


### Problem 1: Manual 1D Convolution

Given input $\vec{x} = [3, 1, 4, 1, 5, 9, 2, 6]$ and kernel $\vec{w} = [1, 0, -1]$ with stride 1 and valid padding:

**(a)** How many output elements will this convolution produce? Show your calculation using the output size formula.

**(b)** Compute all output values by hand.

**(c)** What kind of local pattern does this kernel detect? Explain by examining the structure of the weights.

**(d)** At which output index is the absolute value largest? What does this tell you about the input at that location?


### Problem 2: Stride and Padding Effects

Consider input $\vec{x} = [2, 5, 3, 8, 1, 7, 4, 6, 9, 0]$ (size $n = 10$) and kernel $\vec{w} = [\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}]$ (size $k = 4$).

**(a)** Compute the output size for each combination:

| | Valid padding ($p=0$) | Zero padding ($p=2$) |
|---|---|---|
| Stride $s=1$ | ? | ? |
| Stride $s=2$ | ? | ? |
| Stride $s=3$ | ? | ? |

**(b)** For stride $s=2$, valid padding, compute the first three output values.

**(c)** Describe the physical effect of this kernel on the input signal. Why might this be useful in a noisy environment?

### Problem 3: 1D Convolution as Matrix Multiplication

For input $\vec{x} = [x_0, x_1, x_2, x_3]$ and kernel $\vec{w} = [w_0, w_1]$ with stride 1 and valid padding:

**(a)** Write the output vector $\vec{y}$ as a matrix-vector product $\vec{y} = W\vec{x}$. Explicitly write out the weight matrix $W$.

**(b)** What are the dimensions of $W$?

**(c)** Verify that $W$ is sparse. What fraction of entries in $W$ are zero?

**(d)** Now write the weight matrix for the same kernel but with stride 2. What changes?


### Problem 4: Manual 2D Convolution

Given the $4 \times 4$ input image and $2 \times 2$ kernel:

$$X = \begin{bmatrix} 10 & 20 & 30 & 40 \\ 50 & 60 & 70 & 80 \\ 90 & 100 & 110 & 120 \\ 130 & 140 & 150 & 160 \end{bmatrix}, \qquad W = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

**(a)** What is the output size with stride $[1,1]$ and valid padding?

**(b)** Compute the full output matrix.

**(c)** Interpret the kernel: what local pattern does $W = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$ detect? Where in the output do you see the strongest response, and why?

### Problem 5: Image Smoothing vs. Edge Detection

Consider a $5 \times 5$ grayscale image:

$$X = \begin{bmatrix} 100 & 100 & 100 & 100 & 100 \\ 100 & 100 & 100 & 100 & 100 \\ 10 & 10 & 10 & 100 & 100 \\ 10 & 10 & 10 & 100 & 100 \\ 10 & 10 & 10 & 10 & 10 \end{bmatrix}$$

**(a)** Apply the smoothing kernel $W_s = \frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$ with stride $[1,1]$, valid padding. Compute the output at positions $(0,0)$, $(1,1)$, and $(2,2)$.

**(b)** Apply the vertical edge detection kernel $W_e = \begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}$ with stride $[1,1]$, valid padding. Compute the output at positions $(1,1)$, $(1,2)$, and $(2,1)$.

**(c)** Which kernel produces larger responses at the boundary between the bright (100) and dark (10) regions? Explain why.

### Problem 6: Output Size in Multiple Dimensions

A color image has dimensions $[H, W] = [224, 224]$ with $C = 3$ input channels.

**(a)** A Conv2d layer has kernel size $[3, 3]$, stride $[1, 1]$, and padding $p = 1$. What is the spatial output size?

**(b)** A second Conv2d layer has kernel size $[5, 5]$, stride $[2, 2]$, and valid padding ($p = 0$). Applied to the output of (a), what is the new spatial output size?

**(c)** A third Conv2d layer has kernel size $[3, 3]$, stride $[2, 2]$, padding $p = 1$. Applied to the output of (b), what is the new spatial output size?

**(d)** If each layer has 64 output channels, what is the total number of parameters (weights only, no bias) in the three layers combined? Assume 3 input channels for layer 1, and 64 for layers 2 and 3.

### Problem 7: 3D Convolution for Motion Detection

A video has 8 grayscale frames of size $16 \times 16$. We apply a 3D convolution with:
- Kernel size $[k_T, k_H, k_W] = [2, 3, 3]$
- Stride $[1, 1, 1]$
- Valid padding

**(a)** What is the output size $[o_T, o_H, o_W]$?

**(b)** A motion detection kernel has $W_{t=0} = -\mathbf{1}_{3 \times 3}$ and $W_{t=1} = +\mathbf{1}_{3 \times 3}$ (where $\mathbf{1}_{3 \times 3}$ is a $3 \times 3$ matrix of ones). At a particular slide stop, the $3 \times 3$ spatial patch in frame $t$ has all values 50 and in frame $t+1$ all values 50 (no motion). What is the output?

**(c)** At another slide stop, the $3 \times 3$ patch in frame $t$ has all values 50 and in frame $t+1$ all values 200. What is the output? What does this indicate?

### Problem 8: Transposed Convolution

A 1D convolution has kernel $\vec{w} = [w_0, w_1, w_2]$, input size $n = 5$, stride 1, valid padding.

**(a)** Write the $3 \times 5$ weight matrix $W$ for the forward convolution.

**(b)** Write the $5 \times 3$ transposed weight matrix $W^T$.

**(c)** Given output $\vec{y} = [2, -1, 3]$, compute $\tilde{x} = W^T \vec{y}$ (in terms of $w_0, w_1, w_2$).

**(d)** Is $\tilde{x} = \vec{x}$ (the original input)? Explain why or why not.

**(e)** Compute the output size of a transposed convolution with input size $n' = 3$, stride $s = 2$, kernel size $k = 3$, and padding $p = 0$.


### Problem 9: Max Pooling vs. Average Pooling

Given the $6 \times 6$ feature map:

$$F = \begin{bmatrix} 1 & 3 & 2 & 8 & 0 & 5 \\ 7 & 4 & 6 & 1 & 3 & 2 \\ 0 & 9 & 5 & 4 & 7 & 1 \\ 3 & 2 & 8 & 6 & 0 & 9 \\ 5 & 1 & 0 & 3 & 8 & 4 \\ 6 & 7 & 2 & 1 & 5 & 3 \end{bmatrix}$$

**(a)** Apply $2 \times 2$ max pooling with stride 2. Write the full $3 \times 3$ output.

**(b)** Apply $2 \times 2$ average pooling with stride 2. Write the full $3 \times 3$ output.

**(c)** Apply $3 \times 3$ max pooling with stride 3. Write the output and state its size.

**(d)** Explain why max pooling is generally preferred over average pooling in classification CNNs.

---

### Problem 10: CNN Architecture Design

You are designing a CNN for classifying $28 \times 28$ grayscale images into 10 classes.

**(a)** Fill in the output dimensions for this architecture:

| Layer | Configuration | Output Size |
|---|---|---|
| Input | — | $1 \times 28 \times 28$ |
| Conv2d | 16 filters, $3 \times 3$, stride 1, pad 1 | ? |
| ReLU | — | ? |
| MaxPool2d | $2 \times 2$, stride 2 | ? |
| Conv2d | 32 filters, $3 \times 3$, stride 1, pad 1 | ? |
| ReLU | — | ? |
| MaxPool2d | $2 \times 2$, stride 2 | ? |
| Flatten | — | ? |
| Linear | 10 outputs | ? |

**(b)** How many learnable parameters (weights + biases) does each Conv2d layer have?

**(c)** How many parameters does the final Linear layer have?

**(d)** Compare the total parameter count to a single fully connected layer that maps the $28 \times 28 = 784$ input directly to 10 classes. What is the ratio?


