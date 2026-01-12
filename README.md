# IME775: Data Driven Modeling and Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Marimo](https://img.shields.io/badge/Marimo-Notebooks-FF6B6B?style=for-the-badge)
![NDSU](https://img.shields.io/badge/NDSU-Graduate-006747?style=for-the-badge)

**A graduate course covering mathematical foundations and architectures of deep learning with applications to data-driven modeling and optimization**

</div>

---

## 📚 Course Information

| | |
|---|---|
| **Credits** | 3 |
| **Prerequisites** | Graduate standing |
| **Instructor** | Harun Pirim, PhD |
| **Office** | ENG 106 |
| **Email** | harun.pirim@ndsu.edu |

## 📖 Primary Textbook

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications.

### Supplementary Textbook

> **Watt, J., Borhani, R., & Katsaggelos, A. K. (2020).** *Machine Learning Refined: Foundations, Algorithms, and Applications* (2nd ed.). Cambridge University Press.

Each weekly folder contains supplementary lecture notes and notebooks from ML Refined (without `-math-dl` suffix).

### Additional Recommended Books

- Joel Grus, *Data Science from Scratch*, 2nd Edition, 2019
- Gareth James et al., *An Introduction to Statistical Learning*, Springer, 2023
- Christoph Molnar, *Interpretable Machine Learning*, 2022 (Free online)
- Serg Masis, *Interpretable Machine Learning with Python*, 2nd Edition, 2023

## 🎯 Learning Outcomes

1. Mathematical foundations of deep learning (linear algebra, calculus, optimization)
2. Neural network architectures and training algorithms
3. Modern deep learning: CNNs, RNNs, Transformers
4. Real world data driven modeling, optimization, and inference

## 📊 Grading

| Component | Weight |
|-----------|--------|
| Assignments | 50% |
| Midterm Exam | 20% |
| Project | 30% |

---

## 📅 Course Schedule

| Week | Topic | Reference | Materials |
|:----:|-------|-----------|-----------|
| 01 | Linear Algebra Foundations for Deep Learning | Ch. 1-2 | [📓 Notebook](week-01/notebook-math-dl.py) \| [📝 Notes](week-01/lecture-notes-math-dl.md) \| [📚 Supp](week-01/lecture-notes.md) |
| 02 | Calculus for Deep Learning | Ch. 3 | [📓 Notebook](week-02/notebook-math-dl.py) \| [📝 Notes](week-02/lecture-notes-math-dl.md) \| [📚 Supp](week-02/lecture-notes.md) |
| 03 | Gradient-Based Optimization Algorithms | Ch. 4 | [📓 Notebook](week-03/notebook-math-dl.py) \| [📝 Notes](week-03/lecture-notes-math-dl.md) \| [📚 Supp](week-03/lecture-notes.md) |
| 04 | Neural Network Foundations: Perceptrons to MLPs | Ch. 5 | [📓 Notebook](week-04/notebook-math-dl.py) \| [📝 Notes](week-04/lecture-notes-math-dl.md) \| [📚 Supp](week-04/lecture-notes.md) |
| 05 | Backpropagation: The Engine of Deep Learning | Ch. 6 | [📓 Notebook](week-05/notebook-math-dl.py) \| [📝 Notes](week-05/lecture-notes-math-dl.md) \| [📚 Supp](week-05/lecture-notes.md) |
| 06 | Regularization and Generalization | Ch. 7 | [📓 Notebook](week-06/notebook-math-dl.py) \| [📝 Notes](week-06/lecture-notes-math-dl.md) \| [📚 Supp](week-06/lecture-notes.md) |
| 07 | Modern Deep Architectures: ResNets & Beyond | Ch. 8 | [📓 Notebook](week-07/notebook-math-dl.py) \| [📝 Notes](week-07/lecture-notes-math-dl.md) \| [📚 Supp](week-07/lecture-notes.md) |
| 08 | Unsupervised Learning & Dimensionality Reduction | — | [📓 Notebook](week-08/notebook.py) \| [📝 Notes](week-08/lecture-notes.md) |
| 09 | Convolutional Neural Networks (CNNs) | Ch. 9 | [📓 Notebook](week-09/notebook-math-dl.py) \| [📝 Notes](week-09/lecture-notes-math-dl.md) \| [📚 Supp](week-09/lecture-notes.md) |
| 10 | Advanced CNN Architectures & Applications | — | [📓 Notebook](week-10/notebook.py) \| [📝 Notes](week-10/lecture-notes.md) |
| 11 | Recurrent Neural Networks: RNNs, LSTM, GRU | Ch. 10 | [📓 Notebook](week-11/notebook-math-dl.py) \| [📝 Notes](week-11/lecture-notes-math-dl.md) \| [📚 Supp](week-11/lecture-notes.md) |
| 12 | Sequence-to-Sequence Models | — | [📓 Notebook](week-12/notebook.py) \| [📝 Notes](week-12/lecture-notes.md) |
| 13 | Attention Mechanisms & Transformers | Ch. 11 | [📓 Notebook](week-13/notebook-math-dl.py) \| [📝 Notes](week-13/lecture-notes-math-dl.md) \| [📚 Supp](week-13/lecture-notes.md) |
| 14 | Student Presentations | — | [📝 Guidelines](week-14/presentation-guidelines.md) |
| 15 | Student Presentations | — | [📝 Guidelines](week-15/presentation-guidelines.md) |

---

## 🛠️ Getting Started

### Prerequisites

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

**Option 1: Google Colab (Recommended - No Setup Required)**

Each notebook includes a "Open in Colab" badge at the top. Simply:
1. Browse to any week folder
2. Click on a `.ipynb` file (e.g., `notebook-math-dl.ipynb`)
3. Click the Colab badge at the top of the notebook
4. Run in your browser with free GPU access!

Quick links:
- Week 1 (DL): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harunpirim/IME775/blob/main/week-01/notebook-math-dl.ipynb)
- Week 1 (ML): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harunpirim/IME775/blob/main/week-01/notebook.ipynb)

**Option 2: Marimo (Local - Interactive)**

```bash
# Install marimo
pip install marimo

# Run a specific notebook
marimo edit week-01/notebook-math-dl.py

# Or run in read-only mode
marimo run week-01/notebook-math-dl.py
```

## 📁 Repository Structure

```
IME775/
├── README.md
├── requirements.txt
├── 2nd_ed/                      # ML Refined chapters (supplementary)
│   ├── chapter_1.pdf
│   └── ...
├── week-01/
│   ├── notebook-math-dl.py      # Primary: Deep learning notebook (Marimo)
│   ├── notebook-math-dl.ipynb   # Primary: Deep learning notebook (Colab/Jupyter) 🔗
│   ├── lecture-notes-math-dl.md # Primary: Deep learning notes
│   ├── notebook.py              # Supplementary: ML Refined notebook (Marimo)
│   ├── notebook.ipynb           # Supplementary: ML Refined notebook (Colab/Jupyter) 🔗
│   └── lecture-notes.md         # Supplementary: ML Refined notes
├── week-02/
│   ├── notebook-math-dl.py
│   ├── lecture-notes-math-dl.md
│   ├── notebook.py
│   └── lecture-notes.md
...
└── week-15/
    └── presentation-guidelines.md
```

## 🔧 Libraries & Tools

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **PyTorch** - Deep learning
- **Matplotlib/Seaborn** - Visualization
- **Marimo** - Interactive notebooks

---

## 📖 Primary Textbook Chapter Overview

### Part I: Mathematical Foundations
- **Chapter 1**: Introduction to Deep Learning
  - Machine learning paradigms, deep learning motivation, historical context
- **Chapter 2**: Linear Algebra for Deep Learning
  - Vectors, matrices, tensors, eigenvalues, SVD, norms, matrix calculus
- **Chapter 3**: Calculus for Deep Learning
  - Derivatives, partial derivatives, gradients, chain rule, Jacobians, Hessians
- **Chapter 4**: Optimization Algorithms
  - Gradient descent, SGD, momentum, AdaGrad, RMSprop, Adam, learning rate schedules

### Part II: Neural Network Fundamentals
- **Chapter 5**: Neural Network Basics
  - Perceptrons, activation functions, multi-layer perceptrons, universal approximation
- **Chapter 6**: Backpropagation
  - Computational graphs, forward/backward pass, automatic differentiation
- **Chapter 7**: Regularization and Generalization
  - Overfitting, L1/L2 regularization, dropout, batch normalization, data augmentation

### Part III: Deep Learning Architectures
- **Chapter 8**: Modern Network Architectures
  - Skip connections, ResNets, DenseNets, squeeze-and-excitation, efficient networks
- **Chapter 9**: Convolutional Neural Networks
  - Convolution operation, pooling, padding, stride, LeNet, AlexNet, VGG, receptive fields
- **Chapter 10**: Recurrent Neural Networks
  - Sequence modeling, vanilla RNN, LSTM, GRU, bidirectional RNNs, seq2seq
- **Chapter 11**: Attention and Transformers
  - Attention mechanism, self-attention, multi-head attention, positional encoding, BERT, GPT

---

## 📚 Supplementary Materials (ML Refined)

Based on *Machine Learning Refined* (Watt et al., 2020), additional materials cover classical ML foundations:

| Topic | Week | ML Refined Chapter |
|-------|:----:|:------------------:|
| Introduction to Machine Learning | 01 | Ch. 1 |
| Zero-Order Optimization | 02 | Ch. 2 |
| Gradient Descent | 03 | Ch. 3 |
| Newton's Method | 04 | Ch. 4 |
| Linear Regression | 05 | Ch. 5 |
| Binary Classification | 06 | Ch. 6 |
| Multi-Class Classification | 07 | Ch. 7 |
| PCA & Unsupervised Learning | 08 | Ch. 8 |
| Feature Engineering | 09 | Ch. 9 |
| Nonlinear Features | 10 | Ch. 10 |
| Feature Learning | 11 | Ch. 11 |
| Kernel Methods | 12 | Ch. 12 |
| Tree-Based Methods | 13 | Ch. 14 |

---

## 📜 Course Policies

### Attendance
Attendance in classes is expected per NDSU Policy 333.

### Academic Honesty
NDSU Policy 335: Code of Academic Responsibility and Conduct applies. See [www.ndsu.edu/academichonesty](https://www.ndsu.edu/academichonesty).

### Disability Services
Students with disabilities are invited to contact [Disability Services](https://www.ndsu.edu/disabilityservices).

---

<div align="center">

**North Dakota State University** | Industrial & Manufacturing Systems Engineering

</div>
