# Week 1: Introduction to Machine Learning

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined: Foundations, Algorithms, and Applications* (2nd ed.). Cambridge University Press. **Chapter 1: Introduction to Machine Learning**.

---

## Overview

This week introduces the fundamental concepts of machine learning, its taxonomy, and the connection to mathematical optimization.

---

## Learning Objectives

- Define machine learning and its role in data-driven decision making
- Understand the basic taxonomy of ML problems
- Connect machine learning to mathematical optimization
- Distinguish between regression and classification problems

---

## 1.1 Introduction

Machine learning is a collection of pattern-finding algorithms designed to identify system rules empirically by leveraging data and computing power.

### Historical Context

| Era | Approach | Limitation |
|-----|----------|------------|
| Pre-digital | Philosophical/visual | Limited data |
| Early computing | Statistical models | Limited compute |
| Modern | Machine learning | Abundant data + compute |

### Why Machine Learning Now?

1. **Data abundance**: Sensors, internet, digital systems
2. **Computing power**: GPUs, cloud computing
3. **Algorithm advances**: Deep learning, optimization methods

---

## 1.2 Distinguishing Cats from Dogs: A Machine Learning Approach

The textbook's motivating example illustrates the ML workflow:

### The Problem

Given images of cats and dogs, automatically classify new images.

### The ML Approach

1. **Collect labeled data**: $(image_i, label_i)$ pairs
2. **Extract features**: Convert images to numerical vectors
3. **Choose a model**: Linear classifier, neural network, etc.
4. **Train**: Find parameters that minimize classification error
5. **Predict**: Apply learned model to new images

### Key Insight

> "We don't write explicit rules; we learn them from data."

---

## 1.3 The Basic Taxonomy of Machine Learning Problems

### Supervised Learning

**Definition**: Learning a mapping from inputs to outputs using labeled training data.

$$\text{Given}: \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$$
$$\text{Learn}: f: x \rightarrow y$$

#### Regression

- **Output**: Continuous values
- **Examples**: 
  - Predicting house prices
  - Forecasting stock prices
  - Estimating patient recovery time

#### Classification

- **Output**: Discrete categories
- **Binary**: Two classes (spam/not spam)
- **Multi-class**: Multiple classes (digit recognition: 0-9)

### Unsupervised Learning

**Definition**: Finding structure in unlabeled data.

$$\text{Given}: \{x_1, x_2, \ldots, x_n\}$$
$$\text{Find}: \text{patterns, clusters, representations}$$

#### Clustering

Group similar data points together.

#### Dimensionality Reduction

Compress high-dimensional data to fewer dimensions while preserving structure.

#### Anomaly Detection

Identify unusual data points.

---

## 1.4 Mathematical Optimization

### The Core Connection

> "Nearly every machine learning model is trained by solving an optimization problem."

### The Learning Problem as Optimization

$$\theta^* = \arg\min_{\theta} g(\theta)$$

Where:
- $\theta$: Model parameters
- $g(\theta)$: Cost/loss function
- $\theta^*$: Optimal parameters

### Common Cost Functions

| Problem | Cost Function | Formula |
|---------|--------------|---------|
| Regression | Mean Squared Error | $\frac{1}{n}\sum_i(y_i - f(x_i))^2$ |
| Classification | Cross-Entropy | $-\sum_i y_i \log(\hat{y}_i)$ |

### The Optimization Perspective

Machine learning algorithms are distinguished by:
1. **Model architecture**: What form does $f_\theta(x)$ take?
2. **Cost function**: What are we minimizing?
3. **Optimization method**: How do we find $\theta^*$?

---

## Key Concepts from Chapter 1

### Feature Representation

Raw data must be converted to numerical features:
- **Images**: Pixel values, edge detectors, learned features
- **Text**: Word counts, TF-IDF, embeddings
- **Tabular**: Standardized numerical columns

### The Model

A parameterized function $f_\theta(x)$ that maps inputs to outputs:
- **Linear**: $f(x) = w^Tx + b$
- **Polynomial**: $f(x) = \sum_i w_i x^i$
- **Neural Network**: Composition of linear + nonlinear functions

### Training vs. Testing

- **Training**: Fit model to training data
- **Testing**: Evaluate on held-out test data
- **Generalization**: Performance on unseen data

---

## Mathematical Formulation

### Regression Example

**Model**: $f(x) = w_0 + w_1 x$

**Cost** (Least Squares):
$$g(w_0, w_1) = \frac{1}{n}\sum_{i=1}^n (y_i - (w_0 + w_1 x_i))^2$$

**Optimization**:
$$w^* = \arg\min_{w} g(w)$$

### Classification Example

**Model**: $P(y=1|x) = \sigma(w^Tx + b)$ where $\sigma$ is sigmoid

**Cost** (Cross-Entropy):
$$g(w, b) = -\frac{1}{n}\sum_i [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

---

## Python Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# Regression
X = np.random.randn(100, 2)
y_reg = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.5

reg_model = LinearRegression()
reg_model.fit(X, y_reg)
print("Regression coefficients:", reg_model.coef_)

# Classification
y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)

clf_model = LogisticRegression()
clf_model.fit(X, y_clf)
print("Classification accuracy:", clf_model.score(X, y_clf))
```

---

## Exercises

### Exercise 1.1 (Section 1.2)
For the cats vs. dogs problem, list five features you might extract from images that could help distinguish between the two classes.

### Exercise 1.2 (Section 1.3)
Classify each problem as regression or classification:
1. Predicting tomorrow's temperature
2. Determining if an email is spam
3. Estimating a patient's blood pressure
4. Recognizing handwritten digits

### Exercise 1.3 (Section 1.4)
Write the optimization problem for fitting a line $y = wx + b$ to data points $\{(x_i, y_i)\}$ using least squares.

---

## Summary

- Machine learning finds rules/patterns from data
- Supervised: Learn from labeled examples (regression, classification)
- Unsupervised: Find structure in unlabeled data
- ML training is fundamentally an optimization problem

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 1**.

### Supplementary Reading
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An Introduction to Statistical Learning* (2nd ed.), Chapter 1.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 1.

---

## Next Week Preview

**Week 2: Zero-Order Optimization Techniques** (Chapter 2)
- Global optimization methods
- Local optimization methods
- Random search
- Coordinate search and descent
