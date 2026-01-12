# Week 7: Linear Multi-Class Classification

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 7: Linear Multi-Class Classification**.

---

## Overview

This week extends binary classification to problems with more than two classes.

---

## Learning Objectives

- Extend binary classification to multiple classes
- Implement one-versus-all (OvA) classification
- Understand the softmax classifier and its cost function
- Apply multi-class quality metrics

---

## 7.1 Introduction

### Multi-Class Classification

Predict label $y \in \{1, 2, \ldots, C\}$ where $C > 2$.

### Approaches

1. **One-versus-All (OvA)**: Train $C$ binary classifiers
2. **One-versus-One (OvO)**: Train $\binom{C}{2}$ binary classifiers
3. **Softmax/Multinomial**: Single multi-output classifier

---

## 7.2 One-versus-All Multi-Class Classification

### Strategy

For each class $c \in \{1, \ldots, C\}$:
1. Create binary problem: Class $c$ (positive) vs all others (negative)
2. Train binary classifier $w_c$

### Prediction

$$\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} w_c^T \tilde{x}$$

Choose the class with the highest decision function value.

### Training

For classifier $c$:
- Convert labels: $\tilde{y}_p = +1$ if $y_p = c$, else $\tilde{y}_p = -1$
- Minimize binary classification cost

### Cost Function

Total cost over all classifiers:

$$g(W) = \sum_{c=1}^{C} g_c(w_c)$$

Where $W = [w_1, \ldots, w_C]$.

### Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_ova(X, y, C):
    classifiers = []
    for c in range(C):
        y_binary = (y == c).astype(int)
        clf = LogisticRegression()
        clf.fit(X, y_binary)
        classifiers.append(clf)
    return classifiers

def predict_ova(classifiers, X):
    scores = np.column_stack([clf.decision_function(X) for clf in classifiers])
    return np.argmax(scores, axis=1)
```

---

## 7.5 The Categorical Cross-Entropy Cost Function

### Softmax Function

For scores $z = [z_1, \ldots, z_C]$:

$$\text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

Properties:
- All outputs in $(0, 1)$
- Sum to 1 (valid probability distribution)
- Differentiable

### Probability Model

$$P(y = c | x) = \frac{e^{w_c^T \tilde{x}}}{\sum_{j=1}^{C} e^{w_j^T \tilde{x}}}$$

### Cross-Entropy Cost

$$g(W) = -\frac{1}{P} \sum_{p=1}^{P} \log P(y = y_p | x_p)$$

$$= -\frac{1}{P} \sum_{p=1}^{P} \left[ w_{y_p}^T \tilde{x}_p - \log\left(\sum_{c=1}^{C} e^{w_c^T \tilde{x}_p}\right) \right]$$

### Gradient

For class $c$:

$$\nabla_{w_c} g = \frac{1}{P} \sum_{p=1}^{P} (P(y=c|x_p) - \mathbb{1}[y_p = c]) \tilde{x}_p$$

### Implementation

```python
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def cross_entropy_cost(W, X, y, C):
    scores = X @ W
    probs = softmax(scores)
    log_probs = np.log(probs[np.arange(len(y)), y])
    return -np.mean(log_probs)

def cross_entropy_gradient(W, X, y, C):
    P = len(y)
    scores = X @ W
    probs = softmax(scores)
    
    # Indicator matrix
    indicator = np.zeros((P, C))
    indicator[np.arange(P), y] = 1
    
    return (1/P) * X.T @ (probs - indicator)
```

---

## 7.6 Classification Quality Metrics

### Confusion Matrix

For $C$ classes, matrix $M$ where $M_{ij}$ = count of true class $i$ predicted as class $j$.

### Per-Class Metrics

For class $c$:
- $TP_c$: True positives for class $c$
- $FP_c$: False positives for class $c$
- $FN_c$: False negatives for class $c$

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

### Averaging Strategies

| Method | Precision | Use Case |
|--------|-----------|----------|
| **Macro** | $\frac{1}{C}\sum_c \text{Prec}_c$ | Equal class importance |
| **Micro** | $\frac{\sum_c TP_c}{\sum_c TP_c + FP_c}$ | Global performance |
| **Weighted** | $\sum_c \frac{n_c}{N} \text{Prec}_c$ | Account for imbalance |

### Implementation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Full report
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

---

## 7.7 Weighted Multi-Class Classification

For imbalanced classes:

$$g(W) = -\frac{1}{P} \sum_{p=1}^{P} \beta_{y_p} \log P(y = y_p | x_p)$$

Where $\beta_c$ is the weight for class $c$.

**Common strategy**: $\beta_c \propto 1/n_c$ (inverse class frequency).

---

## 7.8 Stochastic and Mini-Batch Learning

For large datasets, use stochastic gradient descent:

**Stochastic**: One sample per update
**Mini-batch**: Small batch of samples per update

Advantages:
- Memory efficient
- Faster iterations
- Can escape local minima

---

## Comparison: OvA vs Softmax

| Aspect | One-vs-All | Softmax |
|--------|------------|---------|
| Training | Independent classifiers | Joint optimization |
| Probabilities | Not calibrated | Calibrated |
| Computation | Parallelizable | More efficient |
| Decision boundaries | Multiple linear | Multiple linear |

---

## Exercises

### Exercise 7.1 (Section 7.2)
Implement OvA classification from scratch for a 3-class problem.

### Exercise 7.2 (Section 7.5)
Derive the gradient of the softmax cross-entropy cost.

### Exercise 7.3 (Section 7.6)
On an imbalanced dataset, compare macro vs weighted F1 scores.

---

## Summary

- OvA trains $C$ binary classifiers independently
- Softmax provides calibrated probabilities
- Multi-class metrics require careful averaging
- Both methods produce linear decision boundaries

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 7**.

### Supplementary Reading
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 4.3.
- Goodfellow, I., et al. (2016). *Deep Learning*, Section 6.2.

---

## Next Week Preview

**Week 8: Linear Unsupervised Learning & PCA** (Chapter 8)
- Principal Component Analysis
- K-Means Clustering
- Recommender Systems
