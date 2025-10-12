import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Beta
import math

# 3D Distribution Functions - Creating information bottleneck for VAE
def spiral_distribution_3d(n_samples):
    """3D spiral with height variation"""
    t = torch.linspace(0, 4*math.pi, n_samples)
    noise = torch.randn(n_samples) * 0.1
    x = (0.5 * t * torch.cos(t) + noise)
    y = (0.5 * t * torch.sin(t) + noise)
    # Add spiraling height component
    z = 0.2 * t + 0.1 * torch.sin(2*t) + torch.randn(n_samples) * 0.05
    return torch.stack([x, y, z], dim=1)

def gaussian_mixture_4_3d(n_samples):
    """4-Gaussian mixture in 3D space"""
    x1 = torch.randn(n_samples//4, 3) * 0.2 + torch.tensor([2.0, 2.0, 1.0])
    x2 = torch.randn(n_samples//4, 3) * 0.5 + torch.tensor([-2.0, -2.0, -1.0])
    x3 = torch.randn(n_samples//4, 3) * 0.3 + torch.tensor([2.0, -2.0, 0.0])
    x4 = torch.randn(n_samples//4, 3) * 0.4 + torch.tensor([-2.0, 2.0, 1.5])
    return torch.cat([x1, x2, x3, x4], dim=0)

def moons_distribution_3d(n_samples):
    """Two moons in 3D with height variation"""
    n_half = n_samples // 2
    
    # First moon
    t1 = torch.linspace(0, math.pi, n_half)
    noise1 = torch.randn(n_half) * 0.1
    x1 = torch.cos(t1) + noise1
    y1 = torch.sin(t1) + noise1
    z1 = 0.5 * torch.sin(2*t1) + torch.randn(n_half) * 0.1
    
    # Second moon (shifted and rotated)
    t2 = torch.linspace(0, math.pi, n_samples - n_half)
    noise2 = torch.randn(n_samples - n_half) * 0.1
    x2 = 1 - torch.cos(t2) + noise2
    y2 = -torch.sin(t2) - 0.5 + noise2
    z2 = -0.5 * torch.sin(2*t2) + torch.randn(n_samples - n_half) * 0.1
    
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    z = torch.cat([z1, z2])
    return torch.stack([x, y, z], dim=1)

def swiss_roll_3d(n_samples):
    """True 3D Swiss roll manifold"""
    t = torch.rand(n_samples) * 3 * math.pi
    height = torch.rand(n_samples) * 2 - 1
    noise = torch.randn(n_samples, 3) * 0.05
    
    x = t * torch.cos(t) + noise[:, 0]
    y = height + noise[:, 1]
    z = t * torch.sin(t) + noise[:, 2]
    
    return torch.stack([x, y, z], dim=1)

# Test 3D distribution
def visualize_3d_distributions(n_samples=2000):
    """Visualize 3D distributions"""
    test_data_3d = spiral_distribution_3d(n_samples)
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(test_data_3d[:, 0], test_data_3d[:, 1], test_data_3d[:, 2], alpha=0.6, s=10)
    ax1.set_title("3D Spiral Distribution")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(132, projection='3d')
    test_gauss_3d = gaussian_mixture_4_3d(n_samples)
    ax2.scatter(test_gauss_3d[:, 0], test_gauss_3d[:, 1], test_gauss_3d[:, 2], alpha=0.6, s=10, c='red')
    ax2.set_title("3D Gaussian Mixture")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3 = fig.add_subplot(133, projection='3d')
    test_swiss_3d = swiss_roll_3d(n_samples)
    ax3.scatter(test_swiss_3d[:, 0], test_swiss_3d[:, 1], test_swiss_3d[:, 2], alpha=0.6, s=10, c='green')
    ax3.set_title("3D Swiss Roll")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()