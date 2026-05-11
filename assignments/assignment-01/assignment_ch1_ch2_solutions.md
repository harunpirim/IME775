# IME775 Assignment 1 Solutions (Chapters 1-2)

---

## Part A: Concepts from Chapter 1 (Short Answer)
1) Sample answer:  
Machine learning builds a parameterized model that maps inputs to outputs, and the parameters are learned from data. Traditional programming specifies step-by-step rules written by a human. In ML, the program structure is fixed but the parameters are estimated to fit observed examples, enabling generalization to new inputs.

2) Model equation:  
`y = w0 * x0 + w1 * x1 + b`  
Decision rule: If y is large positive, the cat runs away; if y is near 0, the cat ignores; if y is negative, the cat approaches and purrs.

3) Examples:  
- Classification: decide whether a document is sports or politics.  
- Regression: predict a house price from features (size, location, crime rate).

---

## Part B: Vectors and Geometry
4) Norms:  
||a|| = sqrt(3^2 + 4^2) = 5  
||b|| = sqrt((-2)^2 + 1^2) = sqrt(5) = 2.236

5) Dot product:  
a dot b = 3*(-2) + 4*(1) = -6 + 4 = -2

6) Angle:  
cos(theta) = (a dot b) / (||a|| * ||b||) = -2 / (5 * 2.236) = -0.1789  
theta = arccos(-0.1789) = 100.305 degrees

7) Orthogonality:  
Not orthogonal because a dot b = -2 (not close to 0).

---

## Part C: Cosine Similarity and Documents
8) Cosine similarities:  
sim(d1, d2) = 0.854  
sim(d1, d3) = 0.046  
sim(d2, d3) = 0.099

9) Most similar pair: d1 and d2.  
Both are high in machine/learning and have zero biology, so their directions are close.

---

## Part D: Linear Combinations and Dependence
10) v = [7, 3] with standard basis:  
alpha1 = 7, alpha2 = 3.

11) {w1, w2, w3} is linearly dependent because w1 = 2*w2.  
One dependence relation: w1 - 2*w2 = 0 (w3 is not needed).

---

## Part E: Matrices and Transposes
12) AB:
```
[[ 3,  1, -1],
 [-1, 18,  4],
 [ 1,  2,  0]]
```

13) BA is defined because B is 2x3 and A is 3x2.  
BA:
```
[[ 8,  7],
 [ 8, 13]]
```

14) (AB)^T = B^T A^T is verified.  
Both are:
```
[[ 3, -1,  1],
 [ 1, 18,  2],
 [-1,  4,  0]]
```

---

## Part F: Linear Transformations and Orthogonality
15a) With theta = pi/4,  
R = [[0.7071, -0.7071], [0.7071, 0.7071]]  
Rx = [0.7071, 0.7071]

15b) R is orthogonal because R^T R = I (off-diagonal terms cancel).

15c) Length preserved:  
||x|| = 1, ||Rx|| = 1.

---

## Part G: Eigenvalues and Eigenvectors
16) Eigenvalues of A are 5 and 0.

17) Eigenvectors:  
- For lambda = 5: eigenvector proportional to [2, 1].  
- For lambda = 0: eigenvector proportional to [1, -2].  
(Any non-zero scalar multiple is correct.)

---

## References
- [Chapter 1 PDF](file:///Users/harunpirim/Documents/Teaching/IME775/book_pdfs/Ch1.pdf)
- [Chapter 2 PDF](file:///Users/harunpirim/Documents/Teaching/IME775/book_pdfs/Ch2.pdf)
