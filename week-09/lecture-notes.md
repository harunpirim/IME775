# Week 9: Feature Engineering and Selection

## Reference

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined* (2nd ed.). Cambridge University Press. **Chapter 9: Feature Engineering and Selection**.

---

## Overview

This week covers feature engineering and selection techniques that prepare data for machine learning models.

---

## Learning Objectives

- Understand the importance of feature engineering
- Apply feature scaling techniques
- Handle missing values appropriately
- Implement feature selection via boosting and regularization

---

## 9.1 Introduction

### Why Feature Engineering?

> "Features are the backbone of machine learning models."

Good features:
- Capture relevant information
- Are in appropriate scales
- Have no missing values
- Are not redundant

### Feature Engineering Pipeline

```
Raw Data → Feature Extraction → Scaling → Selection → Model
```

---

## 9.2 Histogram Features

### Categorical Encoding

**One-Hot Encoding**: Convert categorical variable to binary vectors.

| Category | Encoded |
|----------|---------|
| Red | [1, 0, 0] |
| Green | [0, 1, 0] |
| Blue | [0, 0, 1] |

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X_categorical)
```

### Binning Continuous Features

Convert continuous variables to discrete bins:

```python
# Age binning
bins = [0, 18, 65, 100]
labels = ['child', 'adult', 'senior']
age_binned = pd.cut(age, bins=bins, labels=labels)
```

---

## 9.3 Feature Scaling via Standard Normalization

### Why Scale?

1. **Gradient descent**: Converges faster with scaled features
2. **Regularization**: Penalizes fairly across features
3. **Distance-based methods**: KNN, K-Means, SVM

### Standard Normalization (Z-score)

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

Where:
- $\mu_j$: Mean of feature $j$
- $\sigma_j$: Standard deviation of feature $j$

**Result**: $\mathbb{E}[\tilde{x}_j] = 0$, $\text{Var}[\tilde{x}_j] = 1$

### Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test
X_test_scaled = scaler.transform(X_test)
```

### Other Scaling Methods

| Method | Formula | Range |
|--------|---------|-------|
| Min-Max | $\frac{x - \min}{\max - \min}$ | [0, 1] |
| Max-Abs | $\frac{x}{\max|x|}$ | [-1, 1] |
| Robust | $\frac{x - \text{median}}{\text{IQR}}$ | Varies |

---

## 9.4 Imputing Missing Values in a Dataset

### Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Mean** | Replace with mean | Numerical, few missing |
| **Median** | Replace with median | Numerical, outliers |
| **Mode** | Replace with most frequent | Categorical |
| **Constant** | Replace with fixed value | When missingness is informative |
| **KNN** | Use k nearest neighbors | Complex patterns |

### Implementation

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
mean_imputer = SimpleImputer(strategy='mean')
X_imputed = mean_imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

### Best Practices

1. Understand why data is missing
2. Consider creating "missing" indicator features
3. Never impute target variable
4. Fit imputer on training data only

---

## 9.5 Feature Scaling via PCA-Sphering

### Whitening (Sphering)

Transform data to have identity covariance:

$$\tilde{X} = (X - \mu) \cdot V \Lambda^{-1/2}$$

Where $\Sigma = V \Lambda V^T$ is the eigendecomposition.

### Effect

- Zero mean
- Unit variance in all directions
- No correlations between features

### Implementation

```python
from sklearn.decomposition import PCA

pca = PCA(whiten=True)
X_whitened = pca.fit_transform(X)
```

---

## 9.6 Feature Selection via Boosting

### Forward Selection

```python
def forward_selection(X, y, model, metric, max_features):
    selected = []
    remaining = list(range(X.shape[1]))
    
    for _ in range(max_features):
        best_score = -np.inf
        best_feature = None
        
        for f in remaining:
            features = selected + [f]
            score = cross_val_score(model, X[:, features], y).mean()
            
            if score > best_score:
                best_score = score
                best_feature = f
        
        selected.append(best_feature)
        remaining.remove(best_feature)
    
    return selected
```

### Backward Elimination

Start with all features, remove one at a time.

### Pros and Cons

| Aspect | Forward | Backward |
|--------|---------|----------|
| Speed | Faster | Slower |
| Starting point | Empty | All features |
| Finds | Local optimum | Local optimum |

---

## 9.7 Feature Selection via Regularization

### L1 Regularization (Lasso)

$$\min_w \frac{1}{P}\|y - Xw\|^2 + \lambda \|w\|_1$$

### Why L1 Gives Sparsity

The L1 ball has corners where coordinates are zero. The solution often lies at these corners.

### Regularization Path

As $\lambda$ increases:
- More coefficients become zero
- Fewer features selected
- Simpler model

### Implementation

```python
from sklearn.linear_model import Lasso, LassoCV

# Cross-validated Lasso
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X, y)

# Selected features
selected = np.where(lasso_cv.coef_ != 0)[0]
print(f"Selected features: {selected}")
print(f"Best alpha: {lasso_cv.alpha_}")
```

### Elastic Net

Combine L1 and L2:

$$\min_w \|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

Benefits:
- Sparsity (from L1)
- Stability (from L2)
- Better with correlated features

---

## Complete Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso

# Define transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline([
    ('preprocess', preprocessor),
    ('lasso', Lasso(alpha=0.1))
])

model.fit(X_train, y_train)
```

---

## Exercises

### Exercise 9.1 (Section 9.3)
Compare model performance with and without feature scaling on a dataset of your choice.

### Exercise 9.2 (Section 9.6)
Implement forward selection and compare selected features with Lasso.

### Exercise 9.3 (Section 9.7)
Plot the Lasso regularization path showing how coefficients change with $\lambda$.

---

## Summary

- Feature engineering is crucial for model performance
- Always scale features before regularized models
- Impute missing values appropriately
- L1 regularization provides automatic feature selection

---

## References

### Primary Text
- Watt, J., Borhani, R., & Katsaggelos, A. K. (2020). *Machine Learning Refined* (2nd ed.), **Chapter 9**.

### Supplementary Reading
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapters 3, 7.
- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*.

---

## Next Week Preview

**Week 10: Principles of Nonlinear Feature Engineering** (Chapter 10)
- Polynomial features
- Nonlinear transformations
- Basis functions
