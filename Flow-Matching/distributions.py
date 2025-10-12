import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Beta
import math

# Target Distribution Functions
def gaussian_mixture_4(n_samples):
    """Original 4-Gaussian mixture"""
    x1 = torch.randn(n_samples//4, 2) * 0.1 + torch.tensor([2.0, 2.0])
    x2 = torch.randn(n_samples//4, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    x3 = torch.randn(n_samples//4, 2) * 0.3 + torch.tensor([2.0, -2.0])
    x4 = torch.randn(n_samples//4, 2) * 0.7 + torch.tensor([-2.0, 2.0])
    return torch.cat([x1, x2, x3, x4], dim=0)

def spiral_distribution(n_samples):
    """Spiral distribution"""
    t = torch.linspace(0, 4*math.pi, n_samples)
    noise = torch.randn(n_samples) * 0.1
    x = (0.5 * t * torch.cos(t) + noise).unsqueeze(1)
    y = (0.5 * t * torch.sin(t) + noise).unsqueeze(1)
    return torch.cat([x, y], dim=1)

def moons_distribution(n_samples):
    """Two moons distribution"""
    n_half = n_samples // 2
    
    # First moon
    t1 = torch.linspace(0, math.pi, n_half)
    noise1 = torch.randn(n_half) * 0.1
    x1 = torch.cos(t1) + noise1
    y1 = torch.sin(t1) + noise1
    
    # Second moon (shifted and rotated)
    t2 = torch.linspace(0, math.pi, n_samples - n_half)
    noise2 = torch.randn(n_samples - n_half) * 0.1
    x2 = 1 - torch.cos(t2) + noise2
    y2 = -torch.sin(t2) - 0.5 + noise2
    
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    return torch.stack([x, y], dim=1)

def circles_distribution(n_samples):
    """Concentric circles"""
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner
    
    # Inner circle
    theta1 = torch.rand(n_inner) * 2 * math.pi
    r1 = 0.5 + torch.randn(n_inner) * 0.1
    x1 = r1 * torch.cos(theta1)
    y1 = r1 * torch.sin(theta1)
    
    # Outer circle
    theta2 = torch.rand(n_outer) * 2 * math.pi
    r2 = 2.0 + torch.randn(n_outer) * 0.1
    x2 = r2 * torch.cos(theta2)
    y2 = r2 * torch.sin(theta2)
    
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    return torch.stack([x, y], dim=1)

def cross_distribution(n_samples):
    """Cross shape distribution"""
    n_quarter = n_samples // 4
    
    # Horizontal bar
    x1 = torch.rand(n_quarter * 2) * 4 - 2  # [-2, 2]
    y1 = torch.randn(n_quarter * 2) * 0.1   # thin horizontal bar
    
    # Vertical bar
    x2 = torch.randn(n_samples - n_quarter * 2) * 0.1  # thin vertical bar
    y2 = torch.rand(n_samples - n_quarter * 2) * 4 - 2  # [-2, 2]
    
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    return torch.stack([x, y], dim=1)

def pinwheel_distribution(n_samples, n_spokes=8):
    """Pinwheel distribution with multiple spokes"""
    samples_per_spoke = n_samples // n_spokes
    remainder = n_samples % n_spokes
    
    all_samples = []
    for i in range(n_spokes):
        n_spoke_samples = samples_per_spoke + (1 if i < remainder else 0)
        
        # Angle for this spoke
        base_angle = 2 * math.pi * i / n_spokes
        
        # Radial distance with some spread
        r = torch.rand(n_spoke_samples) * 2 + 0.5
        # Angle with some spread around the base angle
        theta = base_angle + torch.randn(n_spoke_samples) * 0.2
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        spoke_samples = torch.stack([x, y], dim=1)
        all_samples.append(spoke_samples)
    
    return torch.cat(all_samples, dim=0)

def checkerboard_distribution(n_samples):
    """Checkerboard pattern"""
    # Create a 3x3 grid pattern
    centers = []
    for i in range(3):
        for j in range(3):
            if (i + j) % 2 == 0:  # Only even sum positions
                centers.append([i * 2 - 2, j * 2 - 2])
    
    n_per_center = n_samples // len(centers)
    remainder = n_samples % len(centers)
    
    all_samples = []
    for idx, center in enumerate(centers):
        n_center_samples = n_per_center + (1 if idx < remainder else 0)
        
        center_tensor = torch.tensor(center, dtype=torch.float32)
        samples = torch.randn(n_center_samples, 2) * 0.3 + center_tensor
        all_samples.append(samples)
    
    return torch.cat(all_samples, dim=0)

def swiss_roll_distribution(n_samples):
    """3D Swiss roll projected to 2D"""
    t = torch.rand(n_samples) * 3 * math.pi
    height = torch.rand(n_samples) * 2 - 1
    
    x = t * torch.cos(t)
    y = height
    
    return torch.stack([x, y], dim=1)

# Visualization function for target distributions
def visualize_target_distributions(n_samples=2000):
    """Visualize all available target distributions"""
    distributions = {
        'Gaussian Mixture': gaussian_mixture_4,
        'Spiral': spiral_distribution,
        'Two Moons': moons_distribution,
        'Concentric Circles': circles_distribution,
        'Cross': cross_distribution,
        'Pinwheel': pinwheel_distribution,
        'Checkerboard': checkerboard_distribution,
        'Swiss Roll': swiss_roll_distribution
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, (name, dist_func) in enumerate(distributions.items()):
        if idx < len(axes):
            data = dist_func(n_samples)
            axes[idx].scatter(data[:, 0], data[:, 1], alpha=0.6, s=10, c=colors[idx])
            axes[idx].set_title(f"{name} Distribution")
            axes[idx].grid(True)
            axes[idx].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

# Usage example function
# def test_distribution_with_flow_matching(target_dist_func, dist_name="Custom"):
#     """Test a specific target distribution with flow matching"""
#     print(f"Testing {dist_name} distribution...")
    
#     # Create FlowMatching model with custom target distribution
#     fm = FlowMatching(
#         hidden_dim=64,
#         time_sampling='uniform',
#         time_weighting=False,
#         target_distribution=target_dist_func
#     )
    
#     # Move to device
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     fm.network.to(device)
    
#     # Train the model
#     losses = fm.train(n_epochs=1000)
    
#     # Generate samples and visualize
#     original_data = target_dist_func(2000)
#     samples_ode, _ = fm.sample_ode(n_samples=1000)
#     samples_sde, _ = fm.sample_sde(n_samples=1000)
    
#     # Plot results
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
#     # Original distribution
#     axes[0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=10, c='blue')
#     axes[0].set_title(f"Target: {dist_name}")
#     axes[0].grid(True)
#     axes[0].set_aspect('equal', adjustable='box')
    
#     # Training loss
#     axes[1].plot(losses)
#     axes[1].set_title("Training Loss")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Loss")
#     axes[1].set_yscale('log')
#     axes[1].grid(True)
    
#     # ODE samples
#     axes[2].scatter(samples_ode[:, 0], samples_ode[:, 1], alpha=0.6, s=10, c='red')
#     axes[2].set_title("ODE Samples")
#     axes[2].grid(True)
#     axes[2].set_aspect('equal', adjustable='box')
    
#     # SDE samples
#     axes[3].scatter(samples_sde[:, 0], samples_sde[:, 1], alpha=0.6, s=10, c='green')
#     axes[3].set_title("SDE Samples")
#     axes[3].grid(True)
#     axes[3].set_aspect('equal', adjustable='box')
    
#     plt.tight_layout()
#     plt.show()
    
#     return fm, losses