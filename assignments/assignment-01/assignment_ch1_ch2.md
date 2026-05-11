# IME775 Assignment 1 (Chapters 1-2)
Foundations of machine learning and linear algebra

**Student Name:** ______________________  
**Due Date:** __________________________

## Instructions
- Show your work for all calculations.
- You may use Python for arithmetic, but you must show formulas and steps.
- Round final numeric answers to 3 decimal places unless otherwise stated.
- Submit a single PDF or Markdown file.

---

## Part A: Concepts from Chapter 1 (Short Answer)
1) In 3-5 sentences, define machine learning as described in Chapter 1 and explain how it differs from traditional programming.

2) For the "cat brain" model, write the linear model equation using inputs hardness (x0), sharpness (x1), weights w0, w1, and bias b.  
   Then describe how the output is converted into the three decisions: run away, ignore, approach and purr.

3) Give one example each of:
   - A classification problem
   - A regression (quantitative estimation) problem

---

## Part B: Vectors and Geometry (Chapter 2)
Let a = [3, 4] and b = [-2, 1].

4) Compute the L2 norm of a and b.  
5) Compute the dot product a dot b.  
6) Compute the angle between a and b (in degrees).  
7) Are a and b orthogonal? Explain using a numeric check.

---

## Part C: Cosine Similarity and Documents
Document feature vectors (machine, learning, biology):
- d1 = [5, 2, 0]
- d2 = [3, 4, 0]
- d3 = [0, 1, 8]

8) Compute cosine similarity for all pairs: (d1, d2), (d1, d3), (d2, d3).  
9) Which pair is most similar? Explain in 2-3 sentences.

---

## Part D: Linear Combinations and Dependence
Let v = [7, 3], u1 = [1, 0], u2 = [0, 1].

10) Find scalars $\alpha_1$ and $\alpha_2$ so that $v = \alpha_1*u1 + \alpha_2*u2.$

Let w1 = [2, 4], w2 = [1, 2], w3 = [3, 5].

11) Determine whether {w1, w2, w3} is linearly dependent or independent.  
    Provide a short justification (1-3 sentences) and show a dependence relationship if dependent.

---

## Part E: Matrices and Transposes
Let
```
A = [[ 2, -1],
     [ 3,  4],
     [ 1,  0]]

B = [[ 1,  2, 0],
     [-1,  3, 1]]
```

12) Compute AB.  
13) Is BA defined? If yes, compute BA. If no, explain why.  
14) Verify the transpose property $(AB)^T = B^T A^T$ using your computed matrices.

---

## Part F: Linear Transformations and Orthogonality
15) Consider the rotation matrix for 45 degrees:
```
R = [[cos(\theta), -sin(theta)],
     [sin(theta),  cos(theta)]]
```
with theta = pi/4 and vector x = [1, 0].

15a) Compute Rx.  
15b) Show that R is orthogonal by checking $R^T R = I$.  
15c) Show that the length of x is preserved by rotation.

---

## Part G: Eigenvalues and Eigenvectors
Let
```
A = [[4, 2],
     [2, 1]]
```

16) Compute the eigenvalues of A.  
17) For each eigenvalue, provide one (non-zero) eigenvector.

