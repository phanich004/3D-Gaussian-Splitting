"""
Regularization loss functions for 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class RegularizationLoss(nn.Module):
    """
    Regularization loss functions for 3D Gaussian Splatting.
    This includes sparsity, smoothness, and other regularization terms.
    """
    
    def __init__(self, sparsity_weight: float = 0.01, smoothness_weight: float = 0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute regularization loss
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Total regularization loss
        """
        # Compute individual regularization terms
        sparsity_loss = self._compute_sparsity_loss(gaussians)
        smoothness_loss = self._compute_smoothness_loss(gaussians)
        scale_loss = self._compute_scale_regularization(gaussians)
        opacity_loss = self._compute_opacity_regularization(gaussians)
        
        # Combine losses
        total_loss = (self.sparsity_weight * sparsity_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     scale_loss + opacity_loss)
        
        return total_loss
    
    def _compute_sparsity_loss(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute sparsity loss to encourage sparse Gaussian representation
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Sparsity loss
        """
        opacities = gaussians['opacities']  # (N, 1)
        
        # L1 regularization on opacities (encourage sparsity)
        sparsity_loss = torch.mean(torch.abs(opacities))
        
        # Alternative: entropy-based sparsity
        # entropy_loss = -torch.mean(opacities * torch.log(opacities + 1e-8) + 
        #                          (1 - opacities) * torch.log(1 - opacities + 1e-8))
        
        return sparsity_loss
    
    def _compute_smoothness_loss(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute smoothness loss to encourage smooth Gaussian distribution
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Smoothness loss
        """
        positions = gaussians['positions']  # (N, 3)
        opacities = gaussians['opacities']  # (N, 1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        # Find nearest neighbors
        k = min(10, len(positions) - 1)
        if k <= 0:
            return torch.tensor(0.0, device=positions.device)
        
        # Get k nearest neighbors (excluding self)
        _, nearest_indices = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)  # (N, k+1)
        nearest_indices = nearest_indices[:, 1:]  # (N, k) - exclude self
        
        # Compute smoothness loss based on opacity differences
        smoothness_loss = 0.0
        for i in range(len(positions)):
            neighbor_opacities = opacities[nearest_indices[i]]  # (k, 1)
            current_opacity = opacities[i:i+1]  # (1, 1)
            
            # Opacity smoothness
            opacity_diff = torch.abs(current_opacity - neighbor_opacities)
            smoothness_loss += torch.mean(opacity_diff)
        
        return smoothness_loss / max(1, len(positions))
    
    def _compute_scale_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute scale regularization to prevent extreme scaling
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Scale regularization loss
        """
        scales = gaussians['scales']  # (N, 3)
        
        # Penalize extreme scales
        scale_loss = torch.mean(torch.abs(scales - 1.0))  # Encourage scales around 1.0
        
        # Alternative: log-scale regularization
        # log_scale_loss = torch.mean(torch.abs(torch.log(scales + 1e-8)))
        
        return scale_loss
    
    def _compute_opacity_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute opacity regularization to encourage reasonable opacity values
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Opacity regularization loss
        """
        opacities = gaussians['opacities']  # (N, 1)
        
        # Encourage opacities to be in reasonable range [0, 1]
        # Penalize values too close to 0 or 1
        opacity_loss = torch.mean((opacities - 0.5) ** 2)  # Center around 0.5
        
        return opacity_loss


class GaussianRegularizationLoss(nn.Module):
    """
    Comprehensive regularization for 3D Gaussians including:
    - Position regularization
    - Scale regularization
    - Rotation regularization
    - Color regularization
    """
    
    def __init__(self, position_weight: float = 0.1, scale_weight: float = 0.1,
                 rotation_weight: float = 0.1, color_weight: float = 0.1):
        super().__init__()
        self.position_weight = position_weight
        self.scale_weight = scale_weight
        self.rotation_weight = rotation_weight
        self.color_weight = color_weight
    
    def forward(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute comprehensive Gaussian regularization
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Total regularization loss
        """
        position_loss = self._compute_position_regularization(gaussians)
        scale_loss = self._compute_scale_regularization(gaussians)
        rotation_loss = self._compute_rotation_regularization(gaussians)
        color_loss = self._compute_color_regularization(gaussians)
        
        total_loss = (self.position_weight * position_loss +
                     self.scale_weight * scale_loss +
                     self.rotation_weight * rotation_loss +
                     self.color_weight * color_loss)
        
        return total_loss
    
    def _compute_position_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularize Gaussian positions to prevent clustering
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Position regularization loss
        """
        positions = gaussians['positions']  # (N, 3)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        # Penalize very close Gaussians (encourage separation)
        min_distances = torch.topk(dist_matrix + torch.eye(len(positions), device=positions.device) * 1e6, 
                                 k=2, dim=-1, largest=False)[0][:, 1]  # (N,) - second smallest distance
        
        # Repulsion loss
        repulsion_loss = torch.mean(torch.exp(-min_distances))
        
        return repulsion_loss
    
    def _compute_scale_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularize Gaussian scales
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Scale regularization loss
        """
        scales = gaussians['scales']  # (N, 3)
        
        # Penalize extreme scales
        scale_loss = torch.mean(torch.abs(scales - 1.0))
        
        # Penalize scale anisotropy (encourage isotropic scales)
        scale_variance = torch.var(scales, dim=-1)  # (N,)
        anisotropy_loss = torch.mean(scale_variance)
        
        return scale_loss + anisotropy_loss
    
    def _compute_rotation_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularize Gaussian rotations
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Rotation regularization loss
        """
        rotations = gaussians['rotations']  # (N, 4) - quaternions
        
        # Ensure quaternions are normalized
        quaternion_norms = torch.norm(rotations, dim=-1)  # (N,)
        norm_loss = torch.mean((quaternion_norms - 1.0) ** 2)
        
        return norm_loss
    
    def _compute_color_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularize Gaussian colors
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Color regularization loss
        """
        colors = gaussians['colors']  # (N, 3)
        
        # Encourage colors to be in reasonable range [0, 1]
        color_loss = torch.mean((colors - 0.5) ** 2)
        
        # Encourage color smoothness with neighbors
        positions = gaussians['positions']  # (N, 3)
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        k = min(5, len(positions) - 1)
        if k > 0:
            # Get k nearest neighbors
            _, nearest_indices = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)  # (N, k+1)
            nearest_indices = nearest_indices[:, 1:]  # (N, k) - exclude self
            
            color_smoothness = 0.0
            for i in range(len(positions)):
                neighbor_colors = colors[nearest_indices[i]]  # (k, 3)
                current_color = colors[i:i+1]  # (1, 3)
                
                # Color difference with neighbors
                color_diff = torch.mean(torch.abs(current_color - neighbor_colors))
                color_smoothness += color_diff
            
            color_loss += color_smoothness / max(1, len(positions))
        
        return color_loss


class AdaptiveRegularizationLoss(nn.Module):
    """
    Adaptive regularization that adjusts weights based on training progress
    """
    
    def __init__(self, initial_weight: float = 0.1, decay_rate: float = 0.95):
        super().__init__()
        self.initial_weight = initial_weight
        self.decay_rate = decay_rate
        self.current_weight = initial_weight
        self.step_count = 0
    
    def forward(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute adaptive regularization loss
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Adaptive regularization loss
        """
        # Compute base regularization
        base_loss = self._compute_base_regularization(gaussians)
        
        # Apply adaptive weight
        adaptive_loss = self.current_weight * base_loss
        
        # Update weight for next iteration
        self.step_count += 1
        self.current_weight *= self.decay_rate
        
        return adaptive_loss
    
    def _compute_base_regularization(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute base regularization loss
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            
        Returns:
            torch.Tensor: Base regularization loss
        """
        opacities = gaussians['opacities']  # (N, 1)
        scales = gaussians['scales']  # (N, 3)
        
        # Sparsity loss
        sparsity_loss = torch.mean(torch.abs(opacities))
        
        # Scale regularization
        scale_loss = torch.mean(torch.abs(scales - 1.0))
        
        return sparsity_loss + scale_loss
    
    def reset_weight(self):
        """Reset adaptive weight to initial value"""
        self.current_weight = self.initial_weight
        self.step_count = 0 