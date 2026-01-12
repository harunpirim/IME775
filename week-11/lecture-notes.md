# Week 11: Principles of Feature Learning & Cross-Validation

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 11: Principles of Feature Learning**.

---

## Overview

This week covers cross-validation, regularization, and the principles of model selection.

---

## Learning Objectives

- Understand universal approximation
- Apply cross-validation for model selection
- Implement regularization strategies
- Distinguish training, validation, and test data

---

## 11.1 Introduction

### The Central Problem

Given data, how do we:
1. Choose model architecture?
2. Set hyperparameters?
3. Ensure generalization?

---

## 11.2 Universal Approximators

### Definition

A class of functions is a **universal approximator** if it can approximate any continuous function to arbitrary accuracy.

### Examples

| Class | Description |
|-------|-------------|
| Polynomials | Given enough degree |
| Neural Networks | Given enough hidden units |
| Kernel Methods | With universal kernels |
| Trees | Given enough depth |

### Approximation vs Learning

Universal approximation doesn't guarantee:
- We can **find** the right parameters
- We won't **overfit** to training data
- Good **generalization** to new data

---

## 11.3 Universal Approximation of Real Data

### Finite Data

With finite samples, perfect fit is possible but not desirable.

### The Bias-Complexity Trade-off

- Simple models: High bias, low variance
- Complex models: Low bias, high variance

---

## 11.4 Naive Cross-Validation

### Train-Validation Split

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Model Selection Procedure

1. Define candidate models (e.g., different degrees)
2. For each candidate:
   - Train on training set
   - Evaluate on validation set
3. Select model with lowest validation error
4. Retrain on all training + validation data
5. Final evaluation on test set

### Limitations

- Wastes data (validation set not used for training)
- Variance depends on split
- May not be representative

---

## 11.5 Efficient Cross-Validation via Boosting

### Boosting for Feature Selection

Incrementally add features based on validation performance.

### Forward Stagewise

```python
selected_features = []
remaining = list(range(n_features))

while len(selected_features) < max_features:
    best_score = float('inf')
    best_feature = None
    
    for f in remaining:
        features = selected_features + [f]
        score = validate(X[:, features], y)
        
        if score < best_score:
            best_score = score
            best_feature = f
    
    selected_features.append(best_feature)
    remaining.remove(best_feature)
```

---

## 11.6 Efficient Cross-Validation via Regularization

### Ridge Regression

$$\min_w \|y - Xw\|^2 + \lambda \|w\|_2^2$$

### Cross-Validate λ

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X, y)
print(f"Best alpha: {ridge_cv.alpha_}")
```

### Effect of λ

| λ | Model Behavior |
|---|----------------|
| λ → 0 | No regularization (may overfit) |
| λ small | Slight regularization |
| λ large | Heavy regularization (may underfit) |
| λ → ∞ | All weights → 0 |

---

## 11.7 Testing Data

### The Three Sets

```
Full Data
├── Training Set (60-80%)     # Fit model
├── Validation Set (10-20%)   # Tune hyperparameters
└── Test Set (10-20%)         # Final evaluation
```

### Critical Principles

1. **Never touch test data during development**
2. **Validation data guides model selection**
3. **Test error is the final, honest estimate**

### Common Mistake

Using test data to select hyperparameters → biased estimate → overoptimistic

---

## 11.8 Which Universal Approximator Works Best?

### No Free Lunch

No single model class dominates for all problems.

### Practical Heuristics

| Data Type | Good Starting Points |
|-----------|---------------------|
| Tabular | Gradient boosting, neural nets |
| Images | CNNs |
| Text | Transformers |
| Time series | RNNs, temporal convolutions |

---

## 11.9 Bagging Cross-Validated Models

### Ensemble Methods

Combine multiple models for better performance.

### Bootstrap Aggregating (Bagging)

1. Create multiple bootstrap samples
2. Train model on each
3. Average predictions

### Benefits

- Reduces variance
- Often improves accuracy
- Can estimate uncertainty

---

## 11.10 K-Fold Cross-Validation

### Algorithm

```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mse')
print(f"CV Score: {-scores.mean():.4f} ± {scores.std():.4f}")
```

### Variants

| Method | K | Use Case |
|--------|---|----------|
| 5-Fold | 5 | Default, good balance |
| 10-Fold | 10 | More robust |
| LOOCV | n | Small datasets |
| Stratified | 5-10 | Classification with imbalanced classes |

### Nested Cross-Validation

For hyperparameter tuning + performance estimation:
- Outer loop: Performance estimation
- Inner loop: Hyperparameter selection

---

## 11.11 When Feature Learning Fails

### Failure Modes

1. **Insufficient data**: Model can't learn patterns
2. **Wrong model class**: Can't represent true function
3. **Optimization issues**: Can't find good parameters
4. **Distribution shift**: Test differs from training

### Debugging Strategies

1. Check training error (high = underfitting)
2. Check gap between train/val (large = overfitting)
3. Inspect predictions
4. Try simpler/more complex models

---

## Implementation: Complete CV Pipeline

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('ridge', Ridge())
])

# Define parameter grid
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, 
    cv=5, scoring='neg_mean_squared_error',
    return_train_score=True
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Final evaluation on test set
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {-test_score:.4f}")
```

---

## Exercises

### Exercise 11.1 (Section 11.4)
Implement train-validation split and select the best polynomial degree.

### Exercise 11.2 (Section 11.10)
Compare 5-fold CV with leave-one-out CV on a small dataset.

### Exercise 11.3 (Section 11.6)
Plot the regularization path for Lasso and interpret it.

---

## Summary

- Universal approximators can fit any function
- Cross-validation estimates generalization error
- K-fold is more robust than simple splitting
- Regularization controls complexity continuously
- Test set is only for final evaluation

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 11**.

### Supplementary Reading
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 7.
- Raschka, S. (2018). Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning. *arXiv:1811.12808*.

---

## Next Week Preview

**Week 12: Kernel Methods & Neural Networks** (Chapters 12-13)
- The kernel trick
- Fully connected neural networks
- Activation functions
- Backpropagation
