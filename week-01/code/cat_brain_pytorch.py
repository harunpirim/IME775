"""
Chapter 1: The Cat Brain Model - PyTorch Implementation
=========================================================
From "Math and Architectures of Deep Learning"

This script implements the simple cat brain threat estimator model
and demonstrates the complete machine learning pipeline:
1. Data preparation (normalization)
2. Model architecture definition
3. Training (iterative weight adjustment)
4. Inferencing

Run this code alongside reading the textbook for maximum benefit.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# SECTION 1: DATA PREPARATION
# =============================================================================

def normalize(v, v_min, v_max):
    """
    Equation 1.1: Normalize input to [0, 1] range
    
    v_norm = (v - v_min) / (v_max - v_min)
    
    This maps values from [v_min, v_max] to [0, 1]
    """
    return (v - v_min) / (v_max - v_min)

# Generate training data
# Each sample: (hardness, sharpness) -> threat_score
# Threat score interpretation:
#   - High positive: Run away
#   - Near zero: Ignore
#   - Negative: Approach and purr

def generate_training_data(n_samples=100):
    """
    Generate synthetic training data for the cat brain model.
    
    Ground truth: threat = (hardness + sharpness - 1) / sqrt(2)
    This is the signed distance from the line x0 + x1 = 1
    """
    # Random hardness and sharpness values in [0, 1]
    hardness = np.random.rand(n_samples)
    sharpness = np.random.rand(n_samples)
    
    # Ground truth threat score (Equation 1.4)
    # y = (x0 + x1 - 1) / sqrt(2)
    threat_gt = (hardness + sharpness - 1) / np.sqrt(2)
    
    # Add small noise to make it realistic
    threat_gt += np.random.randn(n_samples) * 0.05
    
    # Stack into input tensor [N, 2] and output tensor [N, 1]
    X = torch.tensor(np.column_stack([hardness, sharpness]), dtype=torch.float32)
    y = torch.tensor(threat_gt.reshape(-1, 1), dtype=torch.float32)
    
    return X, y

# Generate data
X_train, y_train = generate_training_data(200)
X_test, y_test = generate_training_data(50)

print("Training data shape:", X_train.shape)
print("Sample input (hardness, sharpness):", X_train[0].numpy())
print("Sample output (threat score):", y_train[0].item())

# =============================================================================
# SECTION 2: MODEL ARCHITECTURE
# =============================================================================

class CatBrainModel(nn.Module):
    """
    Simple linear model: y = w0*x0 + w1*x1 + b
    
    This is Equation 1.3 from the textbook:
    y(x0, x1) = w0*x0 + w1*x1 + b = w^T * x + b
    
    Parameters:
        w0, w1: weights (how much each input contributes)
        b: bias (constant offset)
    """
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features) creates:
        # - weight matrix of shape [out_features, in_features]
        # - bias vector of shape [out_features]
        self.linear = nn.Linear(2, 1)
        
        # Initialize with random values (as in Algorithm 1.1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.5)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Forward pass: compute y = w^T * x + b
        """
        return self.linear(x)
    
    def get_parameters(self):
        """Return current parameter values for inspection"""
        w = self.linear.weight.detach().numpy().flatten()
        b = self.linear.bias.detach().numpy()[0]
        return w[0], w[1], b

# Create model instance
model = CatBrainModel()
w0, w1, b = model.get_parameters()
print(f"\nInitial parameters: w0={w0:.4f}, w1={w1:.4f}, b={b:.4f}")

# =============================================================================
# SECTION 3: LOSS FUNCTION
# =============================================================================

def compute_loss(y_pred, y_gt):
    """
    Squared error loss (from textbook):
    
    E^2 = sum_i (y_predicted^(i) - y_gt^(i))^2
    
    We use Mean Squared Error (MSE) which is the average:
    MSE = (1/N) * sum_i (y_predicted^(i) - y_gt^(i))^2
    """
    return torch.mean((y_pred - y_gt) ** 2)

# PyTorch provides this as nn.MSELoss()
loss_fn = nn.MSELoss()

# =============================================================================
# SECTION 4: TRAINING (Algorithm 1.1)
# =============================================================================

def train_model(model, X, y, epochs=500, lr=0.1, verbose=True):
    """
    Training loop implementing Algorithm 1.1
    
    1. Initialize parameters with random values (done in __init__)
    2. While error not small enough:
       - For each training instance:
         - Adjust w, b so that E^2 is reduced
    3. Store final parameters as optimal
    
    We use gradient descent optimization (details in later chapters)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {'loss': [], 'w0': [], 'w1': [], 'b': []}
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        y_pred = model(X)
        
        # Compute loss
        loss = loss_fn(y_pred, y)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters (adjust w, b to reduce E^2)
        optimizer.step()
        
        # Record history
        w0, w1, b = model.get_parameters()
        history['loss'].append(loss.item())
        history['w0'].append(w0)
        history['w1'].append(w1)
        history['b'].append(b)
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f}, w0={w0:.4f}, w1={w1:.4f}, b={b:.4f}")
    
    return history

print("\n" + "="*60)
print("TRAINING THE MODEL")
print("="*60)

history = train_model(model, X_train, y_train, epochs=500)

# =============================================================================
# SECTION 5: RESULTS AND ANALYSIS
# =============================================================================

# Final parameters
w0_final, w1_final, b_final = model.get_parameters()
print(f"\nFinal parameters: w0={w0_final:.4f}, w1={w1_final:.4f}, b={b_final:.4f}")

# Theoretical optimal (from Equation 1.4)
w_opt = 1 / np.sqrt(2)
b_opt = -1 / np.sqrt(2)
print(f"Theoretical optimal: w0={w_opt:.4f}, w1={w_opt:.4f}, b={b_opt:.4f}")

# =============================================================================
# SECTION 6: INFERENCING
# =============================================================================

def make_decision(threat_score, threshold=0.2):
    """
    Decision rule from Equation 1.2:
    - y > threshold: Run away (high threat)
    - -threshold <= y <= threshold: Ignore (near zero threat)
    - y < -threshold: Approach and purr (negative threat)
    """
    if threat_score > threshold:
        return "Run away! 🏃"
    elif threat_score < -threshold:
        return "Approach and purr 😻"
    else:
        return "Ignore 😐"

print("\n" + "="*60)
print("INFERENCING ON NEW DATA")
print("="*60)

# Test on some example objects
test_objects = [
    ("Pillow", 0.1, 0.1),
    ("Book", 0.5, 0.5),
    ("Knife", 0.9, 0.95),
    ("Blanket", 0.15, 0.2),
    ("Cactus", 0.85, 0.9),
]

for name, hardness, sharpness in test_objects:
    # Create input tensor
    x = torch.tensor([[hardness, sharpness]], dtype=torch.float32)
    
    # Inference: apply trained model
    with torch.no_grad():
        threat = model(x).item()
    
    decision = make_decision(threat)
    print(f"{name:12} (h={hardness:.2f}, s={sharpness:.2f}) -> threat={threat:+.4f} -> {decision}")

# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================

def plot_training_results(history, X, y, model):
    """Create visualization of training process and results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss over time
    ax1 = axes[0, 0]
    ax1.plot(history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Over Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence
    ax2 = axes[0, 1]
    ax2.plot(history['w0'], 'r-', label='w₀', linewidth=2)
    ax2.plot(history['w1'], 'g-', label='w₁', linewidth=2)
    ax2.plot(history['b'], 'b-', label='b', linewidth=2)
    ax2.axhline(y=1/np.sqrt(2), color='r', linestyle='--', alpha=0.5, label='optimal w')
    ax2.axhline(y=-1/np.sqrt(2), color='b', linestyle='--', alpha=0.5, label='optimal b')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature space with decision boundary
    ax3 = axes[1, 0]
    
    # Get predictions for coloring
    with torch.no_grad():
        predictions = model(X).numpy().flatten()
    
    # Scatter plot of data points
    scatter = ax3.scatter(X[:, 0].numpy(), X[:, 1].numpy(), 
                         c=predictions, cmap='RdYlGn_r', 
                         s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, ax=ax3, label='Threat Score')
    
    # Draw decision boundary (line where y = 0)
    # w0*x0 + w1*x1 + b = 0 -> x1 = (-w0*x0 - b) / w1
    w0, w1, b = model.get_parameters()
    x0_line = np.array([0, 1])
    x1_line = (-w0 * x0_line - b) / w1
    ax3.plot(x0_line, x1_line, 'k-', linewidth=2, label='Decision boundary (y=0)')
    
    # Draw threshold lines
    threshold = 0.2
    x1_upper = (-w0 * x0_line - b + threshold) / w1
    x1_lower = (-w0 * x0_line - b - threshold) / w1
    ax3.plot(x0_line, x1_upper, 'r--', linewidth=1, label=f'y = {threshold}')
    ax3.plot(x0_line, x1_lower, 'g--', linewidth=1, label=f'y = -{threshold}')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('X₀ (Hardness)')
    ax3.set_ylabel('X₁ (Sharpness)')
    ax3.set_title('Feature Space with Decision Boundary')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_aspect('equal')
    
    # Plot 4: Predicted vs Actual
    ax4 = axes[1, 1]
    with torch.no_grad():
        y_pred = model(X).numpy().flatten()
    y_actual = y.numpy().flatten()
    
    ax4.scatter(y_actual, y_pred, alpha=0.5, s=30)
    ax4.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax4.set_xlabel('Actual Threat Score')
    ax4.set_ylabel('Predicted Threat Score')
    ax4.set_title('Predicted vs Actual')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('cat_brain_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as 'cat_brain_training_results.png'")

# Generate the visualization
print("\n" + "="*60)
print("GENERATING VISUALIZATION")
print("="*60)
plot_training_results(history, X_train, y_train, model)

# =============================================================================
# EXERCISES FOR STUDENTS
# =============================================================================
"""
EXERCISES:

1. Parameter Sensitivity:
   - What happens if you initialize weights to very large values?
   - What happens with a very small learning rate (lr=0.001)?
   - What happens with a very large learning rate (lr=10)?

2. Data Quantity:
   - Train with only 10 samples. How do the final parameters compare?
   - Train with 1000 samples. Is there improvement?

3. Model Extension:
   - Add a sigmoid activation: y = sigmoid(w^T x + b)
   - How does this change the decision boundary?

4. Noise Analysis:
   - Increase noise in training data to 0.2, then 0.5
   - How robust is the model to noisy labels?

5. Geometric Interpretation (Section 1.4):
   - Verify that the learned parameters give the signed distance
     from the line x0 + x1 = 1

Try implementing these modifications and observe the results!
"""

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Key Takeaways from this Implementation:

1. MODEL ARCHITECTURE: y = w₀·x₀ + w₁·x₁ + b
   - Simple weighted sum with bias
   - Parameters: w₀, w₁ (weights), b (bias)

2. TRAINING PROCESS:
   - Initialize parameters randomly
   - Iteratively adjust to minimize loss
   - Loss = Mean Squared Error between predictions and ground truth

3. INFERENCING:
   - Apply trained model to new inputs
   - Threshold the output for classification

4. GEOMETRIC INTERPRETATION:
   - Decision boundary is a line in 2D feature space
   - Points above/below the line get different classifications
   
Next: Chapter 2 will introduce the mathematical foundations
(vectors, matrices) needed to understand these operations more deeply.
""")
