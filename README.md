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
| 01 | Vectors, Matrices, and Tensors for Deep Learning | Ch. 1-2 | [📓 Notebook](week-01/notebook-math-dl.py) \| [📝 Notes](week-01/lecture-notes-math-dl.md) \| [📚 Supp](week-01/lecture-notes.md) |
| 02 | Derivatives, Gradients, and the Chain Rule | Ch. 3 | 🔒 Coming Soon |
| 03 | Gradient Descent and Advanced Optimizers (SGD, Adam) | Ch. 4 | 🔒 Coming Soon |
| 04 | Perceptrons, Activation Functions, and MLPs | Ch. 5 | 🔒 Coming Soon |
| 05 | Computational Graphs and Automatic Differentiation | Ch. 6 | 🔒 Coming Soon |
| 06 | Overfitting, Dropout, and Batch Normalization | Ch. 7 | 🔒 Coming Soon |
| 07 | Skip Connections, ResNets, and Efficient Networks | Ch. 8 | 🔒 Coming Soon |
| 08 | Midterm Exam | — | 🔒 Coming Soon |
| 09 | Convolution, Pooling, and CNN Architectures | Ch. 9 | 🔒 Coming Soon |
| 10 | Advanced CNN Applications | — | 🔒 Coming Soon |
| 11 | Sequence Modeling: RNNs, LSTM, and GRU | Ch. 10 | 🔒 Coming Soon |
| 12 | Sequence-to-Sequence and Encoder-Decoder Models | — | 🔒 Coming Soon |
| 13 | Self-Attention, Transformers, BERT, and GPT | Ch. 11 | 🔒 Coming Soon |
| 14 | Student Project Presentations | — | 🔒 Coming Soon |
| 15 | Student Project Presentations | — | 🔒 Coming Soon |

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

All notebooks are provided as **Marimo** interactive notebooks (`.py` files).

```bash
# Install marimo
pip install marimo

# Run a specific notebook in edit mode
marimo edit week-01/notebook-math-dl.py

# Or run in read-only mode
marimo run week-01/notebook-math-dl.py

# Run in presentation mode
marimo run week-01/notebook-math-dl.py --presentation
```

**Why Marimo?**
- 📝 Notebooks are pure Python files (version control friendly)
- 🔄 Reactive execution (cells auto-update when dependencies change)
- 🎨 Clean, modern UI
- 🚀 Fast and lightweight

> **Note:** Course materials are released progressively. Currently, Week 1 materials are available. Additional weeks will be released as the semester progresses.

## 📁 Repository Structure

```
IME775/
├── README.md
├── requirements.txt
├── 2nd_ed/                      # ML Refined chapters (supplementary)
│   ├── chapter_1.pdf
│   └── ...
└── week-01/                     # ✅ Released
    ├── notebook-math-dl.py      # Primary: Deep learning notebook
    ├── lecture-notes-math-dl.md # Primary: Deep learning notes
    ├── notebook.py              # Supplementary: ML Refined notebook
    └── lecture-notes.md         # Supplementary: ML Refined notes

Note: Additional weekly materials (week-02 through week-15) will be released progressively throughout the semester.
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
