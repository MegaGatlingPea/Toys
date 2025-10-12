import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Beta
import math

# used for fm-training process (training)
class TimeSampler:
    
    def __init__(self, mode='uniform'):
        self.mode = mode
    
    def sample(self, batch_size, device='cpu'):
        if self.mode == 'uniform':
            return torch.rand(batch_size, 1, device=device)
        elif self.mode == 'beta':
            # Beta(1.5, 1.5)
            dist = Beta(1.5, 1.5)
            return dist.sample((batch_size, 1)).to(device)
        elif self.mode == 'mix_unif_beta':
            return self._mix_unif_beta_sampling(batch_size, device)
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")
    
    def _mix_unif_beta_sampling(self, batch_size, device, p1=1.5, p2=1.0, p3=0.1):
        beta_dist = Beta(p1, p2)
        beta_samples = beta_dist.sample((batch_size, 1)).to(device)
        uniform_samples = torch.rand(batch_size, 1, device=device)
        selector = torch.rand(batch_size, 1, device=device)
        
        return torch.where(selector < p3, uniform_samples, beta_samples)

# used for fm-sampling process (inference)
class TimeScheduler:
    
    def __init__(self, mode='uniform'):
        self.mode = mode
    
    def get_schedule(self, nsteps, **kwargs):
        if self.mode == 'uniform':
            return torch.linspace(0, 1, nsteps + 1)
        elif self.mode == 'power':
            p = kwargs.get('p', 2.0)
            t = torch.linspace(0, 1, nsteps + 1)
            return t ** p
        elif self.mode == 'log':
            p = kwargs.get('p', 2.0)
            t = 1.0 - torch.logspace(-p, 0, nsteps + 1).flip(0)
            t = t - t.min()
            t = t / t.max()
            return t
        else:
            raise ValueError(f"Unknown schedule mode: {self.mode}")

# Velocity Field -> Score conversion
def vf_to_score(x_t, v, t):
    """Velocity Field -> Score"""
    # s(x_t, t) = (t * v(x_t, t) - x_t) / (1 - t)
    t = t.clamp(min=1e-5, max=1-1e-5) 
    num = t * v - x_t
    den = (1.0 - t)
    return num / den

# Score -> Velocity Field conversion
def score_to_vf(x_t, score, t):
    """Score -> Velocity Field"""
    # v(x_t, t) = (x_t + (1 - t) * s(x_t, t)) / t
    t = t.clamp(min=1e-5, max=1-1e-5)  
    return (x_t + (1.0 - t) * score) / t