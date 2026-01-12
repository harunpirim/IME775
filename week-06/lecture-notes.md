# Week 6: Linear Two-Class Classification

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 6: Linear Two-Class Classification**.

---

## Overview

This week covers binary classification methods including logistic regression, perceptron, and support vector machines.

---

## Learning Objectives

- Understand logistic regression and cross-entropy loss
- Implement the perceptron algorithm
- Formulate support vector machines
- Apply classification quality metrics

---

## 6.1 Introduction

### Binary Classification

Predict a label $y \in \{-1, +1\}$ from features $x$.

### Linear Classifier

$$\hat{y} = \text{sign}(w^T \tilde{x})$$

- Decision boundary: $w^T \tilde{x} = 0$ (hyperplane)
- Positive side: $w^T \tilde{x} > 0 \Rightarrow \hat{y} = +1$
- Negative side: $w^T \tilde{x} < 0 \Rightarrow \hat{y} = -1$

---

## 6.2 Logistic Regression and Cross-Entropy

### The Sigmoid Function

$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

Properties:
- $\sigma(0) = 0.5$
- $\sigma(t) \to 1$ as $t \to \infty$
- $\sigma(t) \to 0$ as $t \to -\infty$
- $\sigma'(t) = \sigma(t)(1 - \sigma(t))$

### Probabilistic Model

$$P(y = +1 | x) = \sigma(w^T \tilde{x})$$

### Cross-Entropy Cost (Softmax Cost)

For labels $y \in \{-1, +1\}$:

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} \log(1 + e^{-y_p w^T \tilde{x}_p})$$

### Gradient

$$\nabla g(w) = -\frac{1}{P} \sum_{p=1}^{P} \frac{y_p \tilde{x}_p}{1 + e^{y_p w^T \tilde{x}_p}}$$

### Implementation

```python
import numpy as np

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def cross_entropy_cost(w, X, y):
    return np.mean(np.log(1 + np.exp(-y * (X @ w))))

def cross_entropy_gradient(w, X, y):
    margins = y * (X @ w)
    return -np.mean((y[:, None] * X) / (1 + np.exp(margins))[:, None], axis=0)

def logistic_regression_gd(X, y, lr=0.1, n_iter=1000):
    w = np.zeros(X.shape[1])
    
    for _ in range(n_iter):
        w = w - lr * cross_entropy_gradient(w, X, y)
    
    return w
```

---

## 6.4 The Perceptron

### The Perceptron Cost

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} \max(0, -y_p w^T \tilde{x}_p)$$

Only misclassified points contribute to the cost.

### The Perceptron Algorithm

```python
def perceptron(X, y, max_iter=100):
    w = np.zeros(X.shape[1])
    
    for _ in range(max_iter):
        misclassified = False
        for i in range(len(y)):
            if y[i] * (X[i] @ w) <= 0:
                w = w + y[i] * X[i]
                misclassified = True
        
        if not misclassified:
            break
    
    return w
```

### Convergence Theorem

If the data is linearly separable, the perceptron converges in finite steps.

### Limitations

- No unique solution
- Doesn't work for non-separable data
- No probability output
- Sensitive to noise

---

## 6.5 Support Vector Machines

### The Margin

The **margin** of a classifier is the distance from the decision boundary to the nearest training point.

$$\text{margin} = \frac{2}{\|w\|}$$

### Hard-Margin SVM

For linearly separable data:

$$\min_{w} \frac{1}{2}\|w\|^2$$

Subject to: $y_p(w^T \tilde{x}_p) \geq 1, \quad p = 1, \ldots, P$

### Soft-Margin SVM

For non-separable data, allow violations:

$$\min_{w,\xi} \frac{1}{2}\|w\|^2 + C \sum_{p=1}^{P} \xi_p$$

Subject to:
- $y_p(w^T \tilde{x}_p) \geq 1 - \xi_p$
- $\xi_p \geq 0$

Where $C > 0$ controls the trade-off between margin and violations.

### Hinge Loss Formulation

Equivalent unconstrained form:

$$g(w) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{P}\sum_{p=1}^{P} \max(0, 1 - y_p w^T \tilde{x}_p)$$

### Support Vectors

Points on or inside the margin boundaries: $y_p(w^T \tilde{x}_p) \leq 1$

### Implementation

```python
from sklearn.svm import SVC

# Linear SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Access support vectors
print("Support vectors:", svm.support_vectors_)
print("Number of SVs:", len(svm.support_))
```

---

## 6.6 Which Approach Works Best?

### Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Logistic** | Probabilistic, smooth | No maximum margin |
| **Perceptron** | Simple, fast | Not unique, no probabilities |
| **SVM** | Maximum margin, sparse | No probabilities, hyperparameter C |

### Practical Advice

- Start with logistic regression (simple, interpretable)
- Use SVM if margin maximization is important
- All perform similarly for most problems

---

## 6.8 Classification Quality Metrics

### Confusion Matrix

|  | Predicted + | Predicted - |
|--|------------|------------|
| **Actual +** | TP | FN |
| **Actual -** | FP | TN |

### Metrics

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** (positive predictive value):
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (sensitivity, true positive rate):
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** (harmonic mean):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Specificity** (true negative rate):
$$\text{Specificity} = \frac{TN}{TN + FP}$$

### ROC Curve

Plot True Positive Rate vs False Positive Rate at different thresholds.

**AUC** (Area Under Curve): Aggregate measure of performance.

### Implementation

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Detailed report
print(classification_report(y_test, y_pred))
```

---

## 6.9 Weighted Classification

For imbalanced classes, weight the cost:

$$g(w) = \frac{1}{P} \sum_{p=1}^{P} \beta_p \cdot \text{loss}(y_p, w^T \tilde{x}_p)$$

**Strategies:**
- Inverse class frequency: $\beta_p = 1/n_{class}$
- Custom weights based on importance

---

## Exercises

### Exercise 6.1 (Section 6.2)
Derive the gradient of the cross-entropy cost function.

### Exercise 6.2 (Section 6.4)
Implement the perceptron algorithm and visualize the decision boundary evolution.

### Exercise 6.3 (Section 6.5)
Compare logistic regression and SVM on a dataset with different C values.

### Exercise 6.4 (Section 6.8)
For an imbalanced dataset, compare accuracy vs F1 score.

---

## Summary

- Logistic regression: probabilistic, uses cross-entropy loss
- Perceptron: simple iterative algorithm
- SVM: maximum margin, uses hinge loss
- Metrics: accuracy, precision, recall, F1

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 6**.

### Supplementary Reading
- James, G., et al. (2023). *An Introduction to Statistical Learning* (2nd ed.), Chapter 4.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.

---

## Next Week Preview

**Week 7: Linear Multi-Class Classification** (Chapter 7)
- One-versus-all classification
- Softmax classifier
- Multi-class metrics
