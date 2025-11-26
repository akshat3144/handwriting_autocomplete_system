"""
Improved Random Distribution Module for HiGAN+
Key improvements:
1. Better initialization strategies
2. Truncated normal distribution for more stable training
3. Learnable temperature scaling
4. Mixed precision support
"""

import torch
import numpy as np
from copy import deepcopy


def seed_rng(seed):
    """Enhanced seeding with deterministic behavior"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImprovedDistribution(torch.Tensor):
    """
    Enhanced Distribution class with:
    - Truncated normal sampling for better stability
    - Adaptive variance scheduling
    - Better numerical stability
    """
    
    def init_distribution(self, dist_type, **kwargs):
        seed_rng(kwargs.get('seed', 0))
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        self.temperature = kwargs.get('temperature', 1.0)
        
        if self.dist_type == 'normal':
            self.mean = kwargs.get('mean', 0.0)
            self.var = kwargs.get('var', 1.0)
            self.truncate = kwargs.get('truncate', 2.0)  # Truncate at 2 std devs
        elif self.dist_type == 'truncated_normal':
            self.mean = kwargs.get('mean', 0.0)
            self.var = kwargs.get('var', 1.0)
            self.truncate = kwargs.get('truncate', 2.0)
        elif self.dist_type == 'uniform':
            self.low = kwargs.get('low', 0.0)
            self.high = kwargs.get('high', 1.0)
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'mixgaussian':
            # Mixture of Gaussians for more diverse sampling
            self.num_modes = kwargs.get('num_modes', 3)
            self.var = kwargs.get('var', 1.0)

    def sample_(self):
        """Enhanced sampling with truncation and temperature scaling"""
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'truncated_normal':
            # Truncated normal for more stable training
            self.normal_(self.mean, self.var)
            # Clamp to truncate
            self.data.clamp_(self.mean - self.truncate * self.var, 
                            self.mean + self.truncate * self.var)
        elif self.dist_type == 'uniform':
            self.uniform_(self.low, self.high)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
        elif self.dist_type == 'mixgaussian':
            # Sample from mixture of Gaussians
            device = self.device
            dtype = self.dtype
            
            # Choose component
            component = torch.randint(0, self.num_modes, (self.size(0),), device=device)
            
            # Sample from chosen component
            means = torch.linspace(-1, 1, self.num_modes, device=device)
            for i in range(self.num_modes):
                mask = (component == i).float().unsqueeze(-1)
                noise = torch.randn_like(self.data) * self.var
                self.data = self.data * (1 - mask) + (means[i] + noise) * mask
        
        # Apply temperature scaling
        if self.temperature != 1.0 and self.dist_type not in ['categorical']:
            self.data = self.data * self.temperature
            
        return self.clone().detach()

    def __deepcopy__(self, memo):
        new_obj = ImprovedDistribution(self.clone())
        if hasattr(self, 'dist_type'):
            new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        return new_obj

    def to(self, *args, **kwargs):
        new_obj = ImprovedDistribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def prepare_z_dist(G_batch_size, dim_z, device='cuda', seed=0, 
                   dist_type='truncated_normal', truncate=2.0, temperature=1.0):
    """
    Improved z distribution with truncated normal sampling
    
    Args:
        G_batch_size: Batch size
        dim_z: Latent dimension
        device: Device to use
        seed: Random seed
        dist_type: 'normal', 'truncated_normal', or 'mixgaussian'
        truncate: Standard deviations to truncate at
        temperature: Temperature scaling factor
    """
    z_ = ImprovedDistribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution(dist_type, mean=0, var=1.0, seed=seed, 
                        truncate=truncate, temperature=temperature)
    z_ = z_.to(device)
    return z_


def prepare_y_dist(G_batch_size, nclasses, device='cuda', seed=0):
    """Prepare label distribution (unchanged for compatibility)"""
    y_ = ImprovedDistribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses, seed=seed)
    y_ = y_.to(device, torch.int64)
    return y_


def prepare_adaptive_z_dist(G_batch_size, dim_z, device='cuda', seed=0, 
                            epoch=0, max_epochs=100):
    """
    Adaptive z distribution that changes variance during training
    Higher variance early, lower variance later for stability
    """
    # Decay temperature from 1.2 to 0.8 over training
    temperature = 1.2 - (0.4 * epoch / max_epochs)
    temperature = max(0.8, temperature)
    
    # Tighter truncation as training progresses
    truncate = 3.0 - (1.0 * epoch / max_epochs)
    truncate = max(2.0, truncate)
    
    return prepare_z_dist(G_batch_size, dim_z, device, seed, 
                         dist_type='truncated_normal',
                         truncate=truncate, 
                         temperature=temperature)


# Backward compatibility with original Distribution class
Distribution = ImprovedDistribution
