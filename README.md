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

Chapter PDFs available in the `2nd_ed/` folder.



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
| 01 | Machine Learning Overview & Vectors, Matrices, Tensors | Ch. 1-2 | [📝 Notes](week-01/Lecture_Notes1.md) \| [📄 PDF](week-01/Lecture_Notes1.pdf) \| [🐱 Cat Brain Demo](week-01/cat_brain_marimo.py) |
| 02 | Classifiers and Vector Calculus (Gradients, Hessians) | Ch. 3 | [📝 Notes](week-02/Lecture_Notes2.md) \| [📄 PDF](week-02/Lecture_Notes2.pdf) \| [📝 Lecture 4](week-02/IME775_Lecture4.md) \| [📄 Lecture 4 PDF](week-02/IME775_Lecture4.pdf) \| [📝 Hyperplanes](week-02/lecture_notes_hyperplanes_ml.md) \| [📄 Hyperplanes PDF](week-02/lecture_notes_hyperplanes_ml.pdf) \| [📓 Notebooks](week-02/) |
| 03 | PCA, SVD, and Dimensionality Reduction | Ch. 4 | 🔒 Coming Soon |
| 04 | Probability Distributions for Machine Learning | Ch. 5 | 🔒 Coming Soon |
| 05 | Bayesian Tools: MLE, MAP, Entropy, KL Divergence | Ch. 6 | 🔒 Coming Soon |
| 06 | Perceptrons, MLPs, and Universal Approximation | Ch. 7 | 🔒 Coming Soon |
| 07 | Forward Propagation and Backpropagation | Ch. 8 | 🔒 Coming Soon |
| 08 | Midterm Exam | — | 🔒 Coming Soon |
| 09 | Loss Functions, Optimization (SGD, Adam), Regularization | Ch. 9 | 🔒 Coming Soon |
| 10 | Convolutions in Neural Networks (1D, 2D, 3D) | Ch. 10 | 🔒 Coming Soon |
| 11 | CNNs: LeNet, VGG, Inception, ResNet | Ch. 11 | 🔒 Coming Soon |
| 12 | Object Detection: R-CNN, Fast/Faster R-CNN | Ch. 11 | 🔒 Coming Soon |
| 13 | Manifolds, Autoencoders, and VAEs | Ch. 12-14 | 🔒 Coming Soon |
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

### Course Materials

Each week's folder contains:
- **Lecture Notes** (`.md`) - Markdown format for easy reading and version control
- **PDF** (`.pdf`) - Printable lecture notes
- **Python Scripts** (`.py`) - Standalone implementations
- **Marimo Notebooks** (`*_marimo.py`) - Interactive demos with widgets

> **Note:** Course materials are released progressively. Currently, Week 1 and Week 2 materials are available. Additional weeks will be released as the semester progresses.

## 📁 Repository Structure

```
IME775/
├── README.md
├── requirements.txt
├── 2nd_ed/                      # ML Refined chapters (supplementary)
│   ├── chapter_1.pdf
│   └── ...
├── week-01/                     # ✅ Released
│   ├── Lecture_Notes1.md        # Lecture notes (Markdown)
│   ├── Lecture_Notes1.pdf       # Lecture notes (PDF)
│   ├── cat_brain_pytorch.py     # Cat Brain model (standalone script)
│   └── cat_brain_marimo.py      # Cat Brain model (interactive notebook)
└── week-02/                     # ✅ Released
    ├── Lecture_Notes2.md        # Lecture notes (Markdown)
    ├── Lecture_Notes2.pdf       # Lecture notes (PDF)
    ├── IME775_Lecture4.md       # Lecture 4 notes (Markdown)
    ├── IME775_Lecture4.pdf      # Lecture 4 notes (PDF)
    ├── lecture_notes_hyperplanes_ml.md  # Hyperplanes notes (Markdown)
    ├── lecture_notes_hyperplanes_ml.pdf  # Hyperplanes notes (PDF)
    ├── IME775_Lecture3-4_Notes.md
    ├── IME775_Lecture3-4_Problems.md
    └── *.py                     # Marimo notebooks and Python scripts

Note: Additional weekly materials (week-03 through week-15) will be released progressively throughout the semester.
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

The Cat Brain demo (`week-01/cat_brain_marimo.py`) implements the threat estimator from Chapter 1 with interactive widgets:

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
marimo edit week-01/cat_brain_marimo.py

# Or run in app mode (output only)
marimo run week-01/cat_brain_marimo.py
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
