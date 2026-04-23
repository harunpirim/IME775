# IME 775 — Lecture 21
## Manifolds, Homeomorphism, and Neural Networks

---

## 1. The Big Picture: Why Do We Care About Manifolds?

In classification, we are given points in some input space and want to separate them by class. The trouble is that **in their original representation, classes are often tangled together in ways no linear boundary can resolve**.

Consider two canonical examples from Chapter 12:

- **1D case:** Class $A \equiv \{-1 \le x \le 1\}$ is "surrounded" by class $B \equiv \{-3 \le x \le -2\} \cup \{2 \le x \le 3\}$. No single threshold separates them.
- **2D case:** Class $A \equiv \{\lVert \vec{x} \rVert_2 \le 1\}$ (a disk) is surrounded by class $B \equiv \{4 \le \lVert \vec{x} \rVert_2^2 \le 9\}$ (an annulus). No straight line separates them.

**Central idea of this chapter:** A neural network does *not* need to find a curved decision boundary in the original space. Instead, each layer **warps the space itself** — it maps the input manifold into a new manifold where the classes *are* linearly separable. The final linear layer then draws a hyperplane in that friendlier space.

> A multilayered neural network is best viewed as a **sequence of continuous, invertible deformations** of the input manifold, progressively untangling the classes until a hyperplane suffices.

To make this precise, we need three tools:
1. A definition of **manifold** (the shape of the data).
2. A definition of **homeomorphism** (the kind of transformation a layer performs).
3. A justification that a neural-network layer *is* a homeomorphism.

---

## 2. Manifolds

A **manifold** is the generalization of a curve, surface, or solid to arbitrary dimensions. Formally, a $d$ -manifold is a set of points satisfying three properties:

1. **Locally Euclidean** (dimension $d$)
2. **Hausdorff**
3. **Second countable**

We examine each below.

### 2.1 Locally Euclidean

**Definition.** A space $M$ is **locally Euclidean of dimension $d$** if every point $p \in M$ has a small neighborhood $U \subset M$ containing $p$ that can be mapped **1:1 to an open subset of $\mathbb{R}^d$ without tearing, twisting, or folding**.

**Physical intuition.**

| Dimension | Example | Local picture |
|---|---|---|
| $d = 1$ | A circle | A short arc can be unrolled (like a piece of string) into a line segment. |
| $d = 2$ | A sphere's surface | A small patch can be flattened (like a rubber membrane) into a planar region. |
| $d = 2$ | A torus (donut) | Same — each small patch flattens into a plane. |
| $d = 3$ | A solid ball | Each small 3D neighborhood deforms into a chunk of $\mathbb{R}^3$. |

**Non-examples.**
- The **figure-8 curve** is *not* a 1-manifold. At the self-intersection, the local neighborhood has four branches arranged like an "X" — no 1:1 map to a line segment is possible.
- An **hourglass surface** is *not* a 2-manifold. The pinch point has neighborhoods that cannot be flattened to a planar disk.

### 2.2 Why "Locally Euclidean" Matters: It Enables Calculus

All of calculus — integration, differentiation, gradients, backpropagation — implicitly assumes we can **approximate curves and surfaces by tiny flat pieces**.

The Riemann integral of a continuous function $f:[a,b]\to\mathbb{R}$ is
$$\int_a^b f(x)\,dx \approx \sum_i f(x_i)\,\Delta x_i,$$
where each term is the area of a tiny rectangle that locally approximates the curve as a straight line. Without the locally Euclidean property, this approximation would fail.

> **Takeaway:** The reason gradients and Taylor expansions even make sense on the complicated surfaces that arise in deep learning is that those surfaces are manifolds.

---

### 2.3 Hausdorff Property

**Definition.** A space is **Hausdorff** if for any two distinct points $p \ne q$, there exist disjoint open neighborhoods $U \ni p$ and $V \ni q$ with $U \cap V = \emptyset$, both lying entirely inside the manifold.

**Intuition.** No matter how close two points on the manifold are, we can always find "wiggle room" around each of them that does not overlap. The real line $\mathbb{R}^1$ is the simplest example: given any two distinct reals, there is space between them to build disjoint intervals.

**Consequence.** Hausdorff spaces are "well-separated" — limits are unique, distinct points stay distinguishable, and standard notions of convergence work as expected.

---

### 2.4 Second Countable Property

Before stating it, three warm-up concepts:

**Open and closed sets.**
- $A \equiv \{0 < x < 1\}$ is **open**: every point has a small neighborhood still inside $A$.
- $A^c \equiv \{0 \le x \le 1\}$ is **closed**: it is $A$ together with its boundary $\{0, 1\}$.
- In 2D, the open disk $S \equiv \{x^2 + y^2 < 1\}$ is open; adding the circle boundary gives the closed disk $S^c \equiv \{x^2 + y^2 \le 1\}$.

**Bounded, compact, precompact.**
- **Bounded** — all points lie within a fixed finite distance of each other.
- **Compact** — bounded *and* closed (e.g., $A^c$ and $S^c$ above).
- **Precompact** — becomes compact by adding its boundary (e.g., the open interval $A$, the open disk $S$).

Note that the unbounded open set $\{-1 < x < \infty\}$ is **open but not precompact** — no boundary can be added to make it bounded.

**Second countable.** A manifold $M$ is **second countable** if it has a countable basis of open sets: there is a countable collection $\mathcal{U} = \{U_i\}$ of precompact open subsets of $M$ such that **every open subset of $M$ can be written as a union of elements of $\mathcal{U}$**.

**Intuition.** We can "cover" any region of the manifold by pasting together finitely or countably many simple, well-behaved patches $U_i$. This is what allows us to do things like **define integrals over a manifold** — we just integrate on each $U_i$ (where things look Euclidean) and stitch the results together.

---

### 2.5 Manifolds With Boundary

Manifolds may or may not include a boundary. A useful pattern:

| $d$-manifold with boundary | Its boundary (a $(d{-}1)$-manifold) |
|---|---|
| Closed disk (2-manifold) | Circle (1-manifold) |
| Closed ball (3-manifold) | Sphere surface (2-manifold) |
| Solid square (2-manifold) | Square outline (1-manifold) |
| Solid cube (3-manifold) | Cube surface (2-manifold) |

> **General rule.** The boundary of a $d$-manifold with boundary is a $(d{-}1)$-manifold.

---

&nbsp;

*Workout 1:* Which of the following are manifolds? For each manifold, state its dimension and whether it has a boundary.

(a) A straight line segment from $(0,0)$ to $(1,1)$.
(b) The letter "T" drawn in the plane.
(c) The surface of a cylinder (a tube with open ends).
(d) Two disjoint circles.

**Solution.**
(a) A 1-manifold with boundary (the two endpoints form a 0-manifold boundary).
(b) **Not a manifold.** At the T-junction, three branches meet — the local neighborhood cannot be mapped 1:1 to a line segment, just like the figure-8.
(c) A 2-manifold with boundary (the two open circles at each end are the boundary, a 1-manifold).
(d) A 1-manifold (being disconnected is fine; each circle is locally Euclidean).

&nbsp;

---

## 3. Homeomorphism

We now formalize the "stretch without tearing" idea.

### 3.1 Definition

A **homeomorphism** between two sets $X$ and $Y$ is a pair of functions $f$ and $f^{-1}$ such that

$$f : X \to Y, \qquad f^{-1} : Y \to X$$

with the following four properties:

1. $f$ is **1:1** (injective): distinct $\vec{x}$ map to distinct $\vec{y}$.
2. $f^{-1}$ is **1:1**: distinct $\vec{y}$ map to distinct $\vec{x}$.
3. $f$ is **continuous**: nearby $\vec{x}$ map to nearby $\vec{y}$.
4. $f^{-1}$ is **continuous**: nearby $\vec{y}$ map to nearby $\vec{x}$.

Spaces related by a homeomorphism are **topologically equivalent** — they have the same "shape" modulo stretching and squishing.

### 3.2 Intuition: What Homeomorphisms Can and Cannot Do

| Allowed | Forbidden |
|---|---|
| Stretch | Cut |
| Squish | Tear |
| Bend | Fold so two distinct points coincide |
| Twist (continuously) | Glue distinct points together |

The cliché: **a coffee cup and a donut are homeomorphic** — both have exactly one hole and one can be deformed into the other without any cutting or gluing.

### 3.3 A Key Property: Preservation of Path-Connectedness

A set is **path-connected** if any two of its points can be joined by a continuous path lying entirely inside the set.

> **Homeomorphisms preserve path-connectedness.** If class $A$ is path-connected before a homeomorphism, it is path-connected after.

This has an immediate consequence for classification. A neural network cannot "break apart" a single path-connected blob of one class into disjoint pieces via its hidden layers — it can only reshape and reposition it. What it *can* do is **move the classes so they no longer surround each other**, as we will now see.

---

## 4. Neural Networks as Homeomorphisms

### 4.1 A Single Linear Layer

Recall from Chapter 8 the form of a single neural-network layer:

$$\vec{z} = f(W\vec{x} + \vec{b})$$

This is the composition of three operations:

1. **Linear map** $\vec{x} \mapsto W\vec{x}$.
2. **Translation** $\vec{u} \mapsto \vec{u} + \vec{b}$.
3. **Nonlinearity** $\vec{v} \mapsto f(\vec{v})$, applied componentwise (e.g., sigmoid, tanh).

Each piece is a continuous, invertible function — provided:
- The weight matrix $W$ is **square with nonzero determinant** (so $W^{-1}$ exists).
- The activation $f$ is **monotonic and continuous** (sigmoid and tanh both qualify; so does leaky ReLU, though plain ReLU is *not* invertible at $0$).

Under these conditions, **each layer is a homeomorphism** from its input manifold to its output manifold. A composition of homeomorphisms is still a homeomorphism, so **the entire multilayer network (up to the final linear classifier) is a single homeomorphism**.

### 4.2 When $W$ Is Not Square

If $W$ is rectangular or has zero determinant, the layer performs a **dimensionality reduction** rather than a homeomorphism. Information is projected, not merely rearranged. This is a deliberate design choice: it can be useful (e.g., for bottleneck layers, PCA-like behavior) but it is *not* a homeomorphism.

### 4.3 Why This View Matters: The Un-Surrounding Principle

Return to the tangled 1D example:
$$A \equiv \{-1 \le x \le 1\}, \qquad B \equiv \{-3 \le x \le -2\} \cup \{2 \le x \le 3\}.$$

On the real line, $A$ is sandwiched between two pieces of $B$ — no single threshold separates them.

A neural network can apply a homeomorphism that **pinches the origin and pulls it up** into a second dimension. In the resulting curved 1-manifold, class $A$ is lifted away from class $B$, and a straight line (the final linear layer) cleanly separates them. Since the transformation is a homeomorphism, the class membership of every point is preserved — we have not "cheated" by creating new points or merging old ones.

The same story repeats in higher dimensions. The 2D disk surrounded by an annulus can be transformed by pinching the origin upward into a 3D bell shape, where a **plane** separates the inner disk from the surrounding ring.

> **Unifying view.** Each layer of a deep network performs a homeomorphic transformation of the data manifold. Successive layers progressively untangle the classes. The final linear layer draws a hyperplane in the now-disentangled manifold.

---

## 5. What Breaks This Picture?

Three honest caveats are worth stating explicitly:

1. **ReLU is not strictly a homeomorphism.** It collapses all negative values to $0$, so $f^{-1}$ does not exist on that half-line. In practice, networks with ReLU still work well — the "homeomorphism" narrative is a useful intuition rather than a rigorous theorem about every modern architecture.
2. **Non-square $W$.** A layer that reduces dimension cannot be inverted. Real networks routinely reduce (and increase) dimension across layers.
3. **Numerical issues.** Even when the math allows inversion, floating-point arithmetic may make $W$ effectively singular.

What survives these caveats is the **conceptual framework**: deep networks deform the input space in structured ways, and the right mental image is one of *topological untangling*, not of finding a complicated decision curve in the original space.

---

## 6. Connection to Earlier Chapters

This chapter ties together ideas we have already developed:

| Concept from earlier chapters | Manifold interpretation |
|---|---|
| **Linear layers** (Ch. 7–8) | Translations + invertible linear maps — homeomorphisms of Euclidean manifolds. |
| **Sigmoid / tanh activations** (Ch. 8) | Smooth, monotonic, invertible — homeomorphisms. |
| **Backpropagation** (Ch. 8) | Chain rule on manifolds; works because manifolds are locally Euclidean. |
| **XOR and MLPs** (Ch. 7) | Cybenko-style universality is the statement that MLPs can realize a wide enough class of homeomorphisms to untangle arbitrary classes. |
| **Convolutional layers** (Ch. 10) | Special structured linear maps — still homeomorphisms when stride/padding/kernel preserve invertibility. |

---

## 7. Summary

| Property | Statement | Intuition |
|---|---|---|
| Locally Euclidean | Every point has a neighborhood mapping 1:1 to $\mathbb{R}^d$ without tearing. | Tiny patches of a curve / surface / volume look flat up close. |
| Hausdorff | Distinct points admit disjoint neighborhoods. | Nearby points can always be "separated with wiggle room." |
| Second countable | Manifold has a countable basis of precompact open sets. | Manifold can be covered by simple, well-behaved patches. |
| Homeomorphism | 1:1, continuous, with 1:1 continuous inverse. | Deform by stretching/squishing; no cutting or gluing. |
| Neural-network layer | (Under mild conditions) a homeomorphism. | A layer reshapes the data manifold without scrambling class membership. |
| Deep network | Composition of homeomorphisms + a linear separator. | Untangle classes step by step, then draw a hyperplane. |

**Key sentence to remember:**

> A classifier neural network maps the input manifold to an output manifold where the classes become linearly separable — and it does so by stretching and squishing, never by tearing or folding.

---

&nbsp;

*Workout 2:* Consider a two-layer network with a $2 \times 2$ invertible weight matrix in each layer and tanh activations. The input data lies on a 2D plane and consists of two classes: the unit disk $A = \{\lVert \vec{x} \rVert \le 1\}$ and an annulus $B = \{4 \le \lVert \vec{x} \rVert^2 \le 9\}$.

(a) Can any number of such layers make $A$ and $B$ linearly separable *while staying in 2D*?
(b) What would make separation possible?

**Solution.**
(a) **No.** Every layer is a homeomorphism of $\mathbb{R}^2 \to \mathbb{R}^2$, so it preserves path-connectedness and "surroundedness." In 2D, $B$ still encloses $A$ after any sequence of such layers — no straight line in 2D can separate them.
(b) **Move to a higher dimension first.** If a layer lifts the data into $\mathbb{R}^3$ (e.g., via a $3 \times 2$ weight matrix — which is not a homeomorphism but is an embedding), we can pinch the origin upward. In the resulting 3D manifold, a plane cleanly separates $A$ from $B$. This is the practical reason hidden layers in real networks often have **more units than the input** — extra dimensions give the network room to untangle.

&nbsp;

---

## 8. Reading and Further Study

- **Primary:** Chaudhury, *Math and Architectures of Deep Learning*, Chapter 12.
- **Deeper dive (optional):** J. M. Lee, *Introduction to Topological Manifolds*, for a rigorous treatment of the properties introduced here.
- **Visual essay (optional):** C. Olah, *Neural Networks, Manifolds, and Topology* — a well-known blog post that illustrates the untangling view with animations.
