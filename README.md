# IME775: Data Driven Modeling and Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
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

Chapter PDFs available in the `resources/textbooks/` folder.


## 🎯 Learning Outcomes

1. Mathematical foundations (linear algebra, calculus, probability, Bayesian methods)
2. Neural network architectures (perceptrons, MLPs, CNNs) and training algorithms
3. Optimization techniques (SGD, Adam) and regularization methods
4. Computer vision: image classification and object detection
5. Generative models: autoencoders and variational autoencoders

## 📊 Grading

| Component | Weight |
|-----------|--------|
| Assignments | 40% |
| Quizzes | 10% |
| Midterm Exam | 20% |
| Project Presentation | 15% |
| Project Report (Paper/Article) | 15% |

---

## 📅 Course Schedule

| Week | Topic | Reference | Materials |
|:----:|-------|-----------|-----------|
| 01 | Machine Learning Overview | Ch. 1 | [📝 Notes](week-01/notes/Lecture_Notes1.md) \| [📄 PDF](week-01/notes/Lecture_Notes1.pdf) \| [🐱 Cat Brain Demo](week-01/code/cat_brain_marimo.py) |
| 02-03 | Vectors, Matrices, Tensors & Linear Algebra | Ch. 2 | [📝 Notes (Ch 2)](week-02-03/notes/IME775_Lecture3-4_Notes.md) \| [📝 Lecture 3](week-02-03/notes/Lecture_Notes2.md) \| [📝 Lecture 4](week-02-03/notes/IME775_Lecture4.md) \| [📓 Notebooks](week-02-03/code/) |
| 04 | Classifiers and Vector Calculus (Gradients, Hessians) | Ch. 3 | [📝 Notes (Ch 3)](week-04/notes/Chapter3_Lecture_Notes.md) \| [📄 PDF](week-04/notes/Chapter3_Lecture_Notes.pdf) \| [🎮 Visualizations](week-04/visualizations/) |
| 05 | PCA, SVD, and Dimensionality Reduction | Ch. 4 | [📝 Notes (Ch 4)](week-05/notes/Chapter4_Lecture_Notes.md) \| [📝 Notes (Ch 4 pt 2)](week-05/notes/Chapter4_Lecture_Notes2.md) \| [📄 PDF](week-05/notes/Chapter4_Lecture_Notes.pdf) \| [📓 Notebooks](week-05/code/) \| [🎮 Visualizations](https://harunpirim.github.io/IME775/) |
| 06 | Probability Distributions for Machine Learning | Ch. 5 | [📝 Lecture 8](week-06/notes/Chapter5_Lecture_Notes.md) \| [📝 Lecture 9](week-06/notes/Chapter5_Lecture_Notes2.md) \| [📝 Problems](assignments/ch5-practice/Chapter5_Problems.md) \| [🎮 Visualizations](https://harunpirim.github.io/IME775/) |
| 07 | Bayesian Tools: MLE, MAP, Entropy, KL Divergence | Ch. 6 | 🔒 Coming Soon |
| 08 | Midterm Exam | — | 🔒 Coming Soon |
| 09 | Perceptrons, MLPs, and Universal Approximation | Ch. 7 | 🔒 Coming Soon |
| 10 | Forward Propagation and Backpropagation | Ch. 8 | 🔒 Coming Soon |
| 11 | Loss Functions, Optimization (SGD, Adam), Regularization | Ch. 9 | 🔒 Coming Soon |
| 12 | Convolutions in Neural Networks (1D, 2D, 3D) | Ch. 10 | 🔒 Coming Soon |
| 13 | CNNs: LeNet, VGG, Inception, ResNet | Ch. 11 | 🔒 Coming Soon |
| 14 | Object Detection: R-CNN, Fast/Faster R-CNN | Ch. 11 | 🔒 Coming Soon |
| 15 | Manifolds, Autoencoders, and VAEs | Ch. 12-14 | 🔒 Coming Soon |

---

## 📋 Assignments & Quizzes

All graded work and practice problems are in the [`assignments/`](assignments/) folder.

| # | Type | Topic | Files |
|---|------|-------|-------|
| 1 | Assignment | Ch. 1 — ML Overview | [📄 Problem Set](assignments/assignment-01/Ch1_Problem_Set.pdf) |
| 2 | Assignment | Ch. 2 — Linear Algebra | [📝 Problems](assignments/assignment-02/IME775_Lecture3-4_Problems.md) |
| 2 | Assignment | Ch. 3 — Classifiers & Calculus | [📝 Solutions](assignments/assignment-02/Assignment2_Solutions.md) |
| 1 | Quiz | Ch. 3 — Classifiers & Calculus | [📄 Solutions](assignments/quiz-01/Q1_Solutions.pdf) |
| — | Practice | Ch. 5 — Probability | [📝 Problems & Solutions](assignments/ch5-practice/Chapter5_Problems.md) |

---

## 🎮 Interactive Visualizations

Explore mathematical concepts through browser-based interactive demos:

**🔗 [Launch Visualizations](https://harunpirim.github.io/IME775/)**

| Week | Visualizations |
|:----:|----------------|
| 04 | Gradient Descent Animator • Taylor Series • Level Contours • Convexity |
| 05 | Quadratic Forms • Positive Definiteness • Matrix Norms • PCA • SVD • LSA |
| 06 | 1D Gaussian • 2D Gaussian • Distributions Explorer • Sampling Demo |

> *No installation required — runs directly in your browser!*

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

### Course Materials

Each week's folder contains:
- **Lecture Notes** (`.md`) - Markdown format for easy reading and version control
- **PDF** (`.pdf`) - Printable lecture notes
- **Python Scripts** (`.py`) - Standalone implementations
- **Marimo Notebooks** (`*_marimo.py`) - Interactive demos with widgets

> **Note:** Course materials are released progressively.

## 📁 Repository Structure

```
IME775/
├── README.md
├── index.html                   # Interactive visualizations landing page
├── requirements.txt
├── resources/
│   └── textbooks/               # ML Refined chapters (supplementary)
│       ├── chapter_1.pdf
│       └── ...
├── assignments/                 # All assignments, quizzes, and solutions
│   ├── assignment-01/           # Ch. 1 problem set
│   ├── assignment-02/           # Ch. 3 problems & solutions
│   ├── quiz-01/                 # Quiz 1 solutions
│   └── ch5-practice/            # Ch. 5 practice problems & solutions
├── week-01/                     # ✅ Released
│   ├── notes/                   # Lecture notes (Markdown & PDF)
│   ├── code/                    # Python scripts & Marimo notebooks
│   └── assets/                  # Images and auxiliary files
├── week-02-03/                  # ✅ Released
│   ├── notes/                   # Lecture notes
│   └── code/                    # Notebooks & scripts
├── week-04/                     # ✅ Released
│   ├── notes/                   # Lecture notes
│   └── visualizations/         # Interactive HTML demos
├── week-05/                     # ✅ Released
│   ├── notes/                   # Lecture notes (Ch 4)
│   ├── code/                    # Python notebooks (Quadratic Forms, PCA/SVD)
│   └── visualizations/          # Interactive HTML demos
└── week-06/                     # ✅ Released
    ├── notes/                   # Lecture notes (Ch 5)
    ├── code/                    # Python notebooks
    └── visualizations/          # Interactive HTML demos

Note: Additional weekly materials (week-07 through week-15) will be released progressively throughout the semester.
```

## 🔧 Libraries & Tools

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **PyTorch** - Deep learning
- **Matplotlib/Seaborn** - Visualization
- **Marimo** - Interactive notebooks

---

## 🐱 Interactive Demos

This course includes interactive notebooks built with [Marimo](https://marimo.io/) for hands-on learning.

### Week 1: Cat Brain Model

The Cat Brain demo (`week-01/code/cat_brain_marimo.py`) implements the threat estimator from Chapter 1 with interactive widgets:

**Features:**
- 🎲 Adjust random seed, sample sizes, and noise levels
- 📐 Tune learning rate, epochs, and optimizer (SGD/Adam)
- 📊 Real-time training visualization (loss curve, parameter convergence)
- 🔮 Test inference with custom objects
- 🎯 Adjustable decision threshold

**To run:**
```bash
# Activate virtual environment
source venv/bin/activate

# Run in edit mode (see code + output)
marimo edit week-01/code/cat_brain_marimo.py

# Or run in app mode (output only)
marimo run week-01/code/cat_brain_marimo.py
```

---

## 📖 Primary Textbook Chapter Overview

> **Krishnendu Chaudhury. (2024).** *Math and Architectures of Deep Learning*. Manning Publications.

### Part I: Mathematical Foundations (Ch. 1-6)
- **Chapter 1**: Overview of Machine Learning and Deep Learning
  - Paradigm shift, function approximation view, cat brain example, regression vs. classification
- **Chapter 2**: Vectors, Matrices, and Tensors
  - Dot product, matrix multiplication, linear transforms, eigenvalues, eigenvectors, diagonalization, spectral decomposition
- **Chapter 3**: Classifiers and Vector Calculus
  - Decision boundaries, loss functions, gradients, Taylor series, Hessian matrix, convexity
- **Chapter 4**: Linear Algebraic Tools
  - PCA, dimensionality reduction, SVD, low-rank approximation, document retrieval with LSA
- **Chapter 5**: Probability Distributions
  - Random variables, joint/marginal probabilities, Gaussian, binomial, multinomial, Bernoulli, categorical
- **Chapter 6**: Bayesian Tools
  - Bayes' theorem, entropy, cross-entropy, KL divergence, MLE, MAP, Gaussian mixture models

### Part II: Neural Networks (Ch. 7-9)
- **Chapter 7**: Function Approximation with Neural Networks
  - Perceptrons, Heaviside function, hyperplanes, MLPs, XOR problem, Cybenko's universal approximation theorem
- **Chapter 8**: Training Neural Networks
  - Sigmoid/tanh activation, linear layers, forward propagation, backpropagation algorithm, gradient descent
- **Chapter 9**: Loss, Optimization, and Regularization
  - Cross-entropy, softmax, focal loss, hinge loss, SGD, momentum, AdaGrad, RMSprop, Adam, L1/L2 regularization, dropout

### Part III: Computer Vision (Ch. 10-11)
- **Chapter 10**: Convolutions in Neural Networks
  - 1D/2D/3D convolution, smoothing, edge detection, transposed convolution, pooling
- **Chapter 11**: Image Classification and Object Detection
  - LeNet, VGG, Inception, ResNet, R-CNN, Fast R-CNN, Faster R-CNN

### Part IV: Advanced Topics (Ch. 12-14)
- **Chapter 12**: Manifolds and Homeomorphism
  - Manifold properties, Hausdorff, second countable, neural networks as homeomorphisms
- **Chapter 13**: Fully Bayesian Parameter Estimation
  - Prior beliefs, conjugate priors, normal-gamma distribution, Bayesian inference
- **Chapter 14**: Latent Spaces and Generative Models
  - Autoencoders, variational autoencoders (VAEs), ELBO, reparameterization trick

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
