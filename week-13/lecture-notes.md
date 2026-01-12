# Week 13: Tree-Based Learners & Advanced Topics

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 14: Tree-Based Learners**.

---

## Overview

This week covers tree-based learning methods including decision trees, gradient boosting, and random forests.

---

## Learning Objectives

- Understand decision tree construction and splitting criteria
- Apply gradient boosting for improved performance
- Implement random forests
- Compare tree-based methods with other approaches

---

## 14.1 Introduction

### Why Trees?

- Intuitive and interpretable
- No feature scaling needed
- Handle mixed feature types
- Capture nonlinear relationships

### Limitations

- Single trees overfit easily
- High variance
- Axis-aligned boundaries

### Solution: Ensembles

Combine multiple trees to reduce variance and improve accuracy.

---

## 14.2 From Stumps to Deep Trees

### Decision Stump

A tree with a single split:
$$f(x) = \begin{cases} c_L & \text{if } x_j \leq t \\ c_R & \text{if } x_j > t \end{cases}$$

### Deeper Trees

Recursively partition the space with more splits.

### Tree Building (CART Algorithm)

```python
def build_tree(X, y, depth=0, max_depth=None):
    # Stopping criteria
    if stopping_criterion(X, y, depth, max_depth):
        return Leaf(predict_value(y))
    
    # Find best split
    best_feature, best_threshold = find_best_split(X, y)
    
    # Split data
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    # Recursively build subtrees
    left_subtree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    
    return Node(best_feature, best_threshold, left_subtree, right_subtree)
```

---

## 14.3 Regression Trees

### Split Criterion

For each candidate split, compute:
$$\text{Cost} = \sum_{x_i \in R_L} (y_i - \bar{y}_L)^2 + \sum_{x_i \in R_R} (y_i - \bar{y}_R)^2$$

Choose the split that minimizes this cost.

### Finding Best Split

```python
def find_best_split_regression(X, y):
    best_cost = float('inf')
    best_feature = None
    best_threshold = None
    
    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            
            cost = np.sum((left - left.mean())**2) + np.sum((right - right.mean())**2)
            
            if cost < best_cost:
                best_cost = cost
                best_feature = j
                best_threshold = t
    
    return best_feature, best_threshold
```

### Prediction

Traverse tree from root to leaf, return leaf value.

---

## 14.4 Classification Trees

### Split Criteria

**Gini Impurity:**
$$G = \sum_{c=1}^{C} p_c (1 - p_c) = 1 - \sum_{c=1}^{C} p_c^2$$

**Entropy (Information Gain):**
$$H = -\sum_{c=1}^{C} p_c \log_2 p_c$$

**Misclassification Error:**
$$E = 1 - \max_c p_c$$

### Comparison

| Criterion | Sensitivity | Use Case |
|-----------|-------------|----------|
| Gini | Moderate | Default in sklearn |
| Entropy | More sensitive | When information gain matters |
| Misclassification | Least sensitive | Final pruning |

### Weighted Impurity Decrease

$$\Delta I = I_{parent} - \frac{n_L}{n} I_L - \frac{n_R}{n} I_R$$

---

## 14.5 Gradient Boosting

### The Idea

Build trees sequentially, each fitting the residuals:
$$f_m(x) = f_{m-1}(x) + \gamma h_m(x)$$

### Algorithm

```python
def gradient_boosting(X, y, n_estimators, learning_rate, max_depth):
    # Initialize with mean
    f = np.full(len(y), y.mean())
    trees = []
    
    for m in range(n_estimators):
        # Compute residuals
        residuals = y - f
        
        # Fit tree to residuals
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X, residuals)
        trees.append(tree)
        
        # Update predictions
        f = f + learning_rate * tree.predict(X)
    
    return trees, y.mean()
```

### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More = more capacity, risk of overfitting |
| `learning_rate` | Smaller = slower learning, better generalization |
| `max_depth` | Controls individual tree complexity |
| `subsample` | Stochastic gradient boosting |

### XGBoost and LightGBM

Modern implementations with:
- Regularization
- Parallel processing
- Missing value handling
- Categorical feature support

---

## 14.6 Random Forests

### Algorithm

```python
def random_forest(X, y, n_estimators, max_features, max_depth):
    trees = []
    n_samples = len(y)
    
    for b in range(n_estimators):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]
        
        # Train tree with random feature subset
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features
        )
        tree.fit(X_boot, y_boot)
        trees.append(tree)
    
    return trees

def predict_forest(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    # Majority vote
    return np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), 
        axis=0, arr=predictions
    )
```

### Why Random Forests Work

1. **Bootstrap** reduces variance through averaging
2. **Feature randomness** decorrelates trees
3. **Ensemble** is more robust than single tree

### Out-of-Bag (OOB) Error

Use samples not in bootstrap for validation:
- ~37% of samples left out per tree
- Free cross-validation estimate

---

## 14.7 Cross-Validation for Trees

### Pruning

1. Grow full tree
2. Prune back based on validation error
3. Select tree with best validation performance

### Cost-Complexity Pruning

$$R_\alpha(T) = R(T) + \alpha |T|$$

Where:
- $R(T)$: Training error
- $|T|$: Number of leaves
- $\alpha$: Complexity parameter

---

## Implementation: Complete Example

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Random Forest with tuning
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
rf_cv.fit(X_train, y_train)

print(f"Best RF params: {rf_cv.best_params_}")
print(f"Best RF score: {rf_cv.best_score_:.4f}")

# Gradient Boosting with tuning
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb = GradientBoostingClassifier(random_state=42)
gb_cv = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy')
gb_cv.fit(X_train, y_train)

print(f"Best GB params: {gb_cv.best_params_}")
print(f"Best GB score: {gb_cv.best_score_:.4f}")
```

---

## Feature Importance

### Mean Decrease in Impurity

Average impurity decrease weighted by samples reaching node.

```python
# Get feature importance
importances = model.feature_importances_

# Plot
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
```

### Permutation Importance

Measure accuracy decrease when feature is shuffled.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
```

---

## Comparison Summary

| Method | Bias | Variance | Interpretability | Speed |
|--------|------|----------|-----------------|-------|
| Single Tree | Low | High | High | Fast |
| Random Forest | Low | Medium | Low | Parallel |
| Gradient Boosting | Low | Low | Low | Sequential |

---

## Exercises

### Exercise 14.1 (Section 14.3)
Implement a regression tree from scratch for 1D data.

### Exercise 14.2 (Section 14.5)
Compare gradient boosting with different learning rates.

### Exercise 14.3 (Section 14.6)
Analyze the effect of `max_features` on random forest performance.

---

## Summary

- Decision trees partition space with axis-aligned splits
- Single trees are interpretable but overfit
- Random forests reduce variance through bootstrap aggregation
- Gradient boosting reduces bias through sequential fitting
- Modern implementations (XGBoost, LightGBM) are highly optimized

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 14**.

### Supplementary Reading
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapters 9-10.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.

---

## Course Conclusion

This completes the theoretical foundations of IME775. Key takeaways:

1. **Optimization** is the foundation of machine learning
2. **Linear models** are simple but powerful with the right features
3. **Nonlinear methods** (kernels, neural nets, trees) capture complex patterns
4. **Cross-validation** is essential for model selection
5. **Feature engineering** often matters more than model choice

### Weeks 14-15: Student Presentations

Apply these concepts to real-world problems!
