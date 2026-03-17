# Lecture Note: Representation of the “Tower” Network

## 1. Goal

We want a neural network that outputs

$$y = 1$$

only when the input point

$$x = \begin{bmatrix}x_0 \\ x_1\end{bmatrix}$$

lies inside the square

$$-0.5 < x_0 < 0.5, \qquad -0.5 < x_1 < 0.5,$$

and outputs

$$y = 0$$

otherwise.

If we plot the output vertically over the ($x_0$,$x_1$)-plane, this creates a rectangular **tower**.

---

## 2. Step Activation Function

We use the threshold activation

$$\theta(t)= \begin{cases} 1, & t>0 \\ 0, & t\le 0\end{cases}$$

applied elementwise to vectors.

---

## 3. Dimension-Consistent Network

A clean version of the network is a **2 → 4 → 2 → 1** threshold network.

### Layer 1: Four wall tests

Let

$$W^{(1)}=\begin{bmatrix}1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{bmatrix}, \qquad b^{(1)}= \begin{bmatrix} 0.5 \\ 0.5 \\ 0.5 \\ 0.5 \end{bmatrix}.$$

Then

$$z^{(1)} = W^{(1)}x + b^{(1)}= \begin{bmatrix} x_0+0.5 \\ x_0+0.5 \\ x_1+0.5 \\ -x_1+0.5 \end{bmatrix}, \qquad a^{(1)} = \theta(z^{(1)}).$$

So the four neurons test:

- $a_1 = \theta(x_0+0.5)$: checks $x_0 > -0.5$
- $a_2 = \theta(-x_0+0.5)$: checks $x_0 < 0.5$
- $a_3 = \theta(x_1+0.5)$: checks $x_1 > -0.5$
- $a_4 = \theta(-x_1+0.5)$: checks $x_1 < 0.5$

Thus, Layer 1 creates the four boundaries of the square.

---

### Layer 2: Two interval tests

Now combine the four wall tests into two “inside-interval” tests:

$$W^{(2)}=\begin{bmatrix}1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1\end{bmatrix}, \qquad b^{(2)}= \begin{bmatrix}-1.5 \\ -1.5\end{bmatrix}.$$

Then

$$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} = \begin{bmatrix} a_1 + a_2 - 1.5 \\ a_3 + a_4 - 1.5 \end{bmatrix}, \qquad a^{(2)} = \theta(z^{(2)}).$$

Write

$$a^{(2)}= \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}.$$

Then:

- $b_1 = \theta(a_1 + a_2 - 1.5)$ is 1 only if both $a_1$ and $a_2$ are 1.
- $b_2 = \theta(a_3 + a_4 - 1.5)$ is 1 only if both $a_3$ and $a_4$ are 1.

This means:

- $b_1=1$ iff $-0.5 < x_0 < 0.5$
- $b_2=1$ iff $-0.5 < x_1 < 0.5$

### Why is **−1.5** used?

Because for binary inputs $u,v \in \{0,1\}$,

$$\theta(u+v-1.5)=1 \iff u=v=1.$$

So **−1.5 implements a 2-out-of-2 threshold**, which is exactly an AND gate for two binary inputs.

---

### Layer 3: Final inside-square test

Now require both coordinate interval tests to be true:

$$W^{(3)}=\begin{bmatrix}1 & 1\end{bmatrix},\qquad b^{(3)}=-1.5.$$

Then

$$z^{(3)} = W^{(3)}a^{(2)} + b^{(3)} = b_1+b_2-1.5, \qquad y = \theta(z^{(3)}).$$

Therefore,

$$y = 1 \iff b_1=b_2=1.$$

which means

$$y=1 \iff -0.5 < x_0 < 0.5 \text{ and } -0.5 < x_1 < 0.5.$$

This is the corrected mathematical representation of the network.

---

## 4. Full Network in Compact Form

$$a^{(1)} = \theta(W^{(1)}x+b^{(1)}),$$

$$a^{(2)} = \theta(W^{(2)}a^{(1)}+b^{(2)}),$$

$$y = \theta(W^{(3)}a^{(2)}+b^{(3)}).$$

with

$$W^{(1)}=\begin{bmatrix}1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{bmatrix}, \qquad b^{(1)}=\begin{bmatrix}0.5 \\ 0.5 \\ 0.5 \\ 0.5 \end{bmatrix}.$$

$$W^{(2)}=\begin{bmatrix}1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}, \qquad b^{(2)}=\begin{bmatrix}-1.5 \\ -1.5 \end{bmatrix}.$$

$$W^{(3)}=\begin{bmatrix}1 & 1 \end{bmatrix}, \qquad b^{(3)}=-1.5.$$

---

## 5. Layer-by-Layer Interpretation

| Layer | Role | Mathematical meaning |
|---|---|---|
| Layer 1 | Four wall detectors | Tests which side of each boundary line the point lies on |
| Layer 2 | Two interval detectors | Checks whether $x_0$ and $x_1$ are each within their bounds |
| Layer 3 | Final AND | Declares whether the point is inside the square |

So the network works as:

$$(x_0,x_1) \rightarrow \text{4 wall tests} \rightarrow \text{2 interval tests} \rightarrow \text{inside-square output}.$$

---

## 6. Worked Example 1: Point Inside the Square

Take

$$x=\begin{bmatrix}0.2 \\ 0.1\end{bmatrix}.$$   

### Layer 1

$$z^{(1)} = W^{(1)}x + b^{(1)} = \begin{bmatrix} 0.2+0.5 \\ -0.2+0.5 \\ 0.1+0.5 \\ -0.1+0.5 \end{bmatrix} = \begin{bmatrix} 0.7 \\ 0.3 \\ 0.6 \\ 0.4 \end{bmatrix}.$$

so

$$a^{(1)} = \theta(z^{(1)}) = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}.$$ 

### Layer 2

$$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} + \begin{bmatrix} -1.5 \\ -1.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}.$$

so

$$a^{(2)} = \theta(z^{(2)}) = \begin{bmatrix} 1 \\ 1 \end{bmatrix}.$$

### Layer 3

$$z^{(3)} = W^{(3)}a^{(2)} + b^{(3)} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} - 1.5 = 0.5.$$

hence

$$y=\theta(z^{(3)})=1.$$


So the point is **inside** the square.

---

## 7. Worked Example 2: Point Outside the Square

Take

$$x=\begin{bmatrix}0.8 \\ 0.1\end{bmatrix}.$$

### Layer 1

$$z^{(1)} = W^{(1)}x + b^{(1)} = \begin{bmatrix} 0.8+0.5 \\ -0.8+0.5 \\ 0.1+0.5 \\ -0.1+0.5 \end{bmatrix} = \begin{bmatrix} 1.3 \\ -0.3 \\ 0.6 \\ 0.4 \end{bmatrix}.$$  
so
$$a^{(1)} = \theta(z^{(1)}) = \begin{bmatrix} 1 \\ 0 \\ 1 \\ 1 \end{bmatrix}.$$ 

so

$$a^{(2)} = \theta(z^{(2)}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$

### Layer 2

$$z^{(3)} = W^{(3)}a^{(2)} + b^{(3)} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} - 1.5 = -0.5.$$

hence

$$y=\theta(z^{(3)})=0.$$


so

$$a^{(2)} = \theta(z^{(2)}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$

### Layer 3

$$z^{(3)} = W^{(3)}a^{(2)} + b^{(3)} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} - 1.5 = -0.5.$$

hence

$$y=\theta(z^{(3)})=0.$$

So the point is **outside** the square.

---

## 8. Summary Table

| Point | Layer 1 output $a^{(1)}$ | Layer 2 output $a^{(2)}$ | Final output $y$ |
|---|---|---|---|
| $\begin{bmatrix}0.2 \\ 0.1\end{bmatrix}$ | $\begin{bmatrix}1\\1\\1\\1\end{bmatrix}$ | $\begin{bmatrix}1\\1\end{bmatrix}$ | 1 |
| $\begin{bmatrix}0.8 \\ 0.1\end{bmatrix}$ | $\begin{bmatrix}1\\0\\1\\1\end{bmatrix}$ | $\begin{bmatrix}0\\1\end{bmatrix}$ | 0 |

---

## 9. Main Takeaway

This representation shows how layers in a threshold network can be interpreted geometrically and logically:

1. **Layer 1** creates half-space tests.
2. **Layer 2** combines pairs of tests into interval checks using the threshold **−1.5**.
3. **Layer 3** combines the two interval checks into a final inside-square decision.

Thus, multilayer networks can build localized regions in input space, which is the basic idea behind constructing “towers” and, more generally, approximating complicated functions.
