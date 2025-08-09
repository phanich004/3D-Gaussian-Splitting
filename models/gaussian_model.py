"""
3D Gaussian Model for Pose-Free 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GaussianModel(nn.Module):
    """
    3D Gaussian Model that parameterizes 3D Gaussians with:
    - Position (x, y, z)
    - Scale (sx, sy, sz)
    - Rotation (quaternion)
    - Color (RGB)
    - Opacity (alpha)
    """
    
    def __init__(self, num_gaussians: int = 100000, device: torch.device = torch.device('cuda')):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.device = device
        
        # Initialize learnable parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize Gaussian parameters"""
        # Position: (N, 3) - initialized randomly in [-1, 1]
        self.positions = nn.Parameter(
            torch.rand(self.num_gaussians, 3, device=self.device) * 2 - 1
        )
        
        # Scale: (N, 3) - initialized with small positive values
        self.scales = nn.Parameter(
            torch.ones(self.num_gaussians, 3, device=self.device) * 0.01
        )
        
        # Rotation: (N, 4) - quaternions, initialized as identity rotations
        self.rotations = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_gaussians, 1)
        )
        
        # Color: (N, 3) - RGB values, initialized randomly
        self.colors = nn.Parameter(
            torch.rand(self.num_gaussians, 3, device=self.device)
        )
        
        # Opacity: (N, 1) - initialized with moderate opacity
        self.opacities = nn.Parameter(
            torch.ones(self.num_gaussians, 1, device=self.device) * 0.5
        )
    
    def forward(self) -> dict:
        """
        Forward pass returning a dictionary of Gaussian parameters
        
        Returns:
            dict: Dictionary containing:
                - positions: (N, 3) Gaussian positions
                - scales: (N, 3) Gaussian scales
                - rotations: (N, 4) Gaussian rotations (quaternions)
                - colors: (N, 3) Gaussian colors (RGB)
                - opacities: (N, 1) Gaussian opacities
        """
        return {
            'positions': self.positions,
            'scales': F.softplus(self.scales),  # Ensure positive scales
            'rotations': F.normalize(self.rotations, dim=-1),  # Normalize quaternions
            'colors': torch.sigmoid(self.colors),  # Ensure colors in [0, 1]
            'opacities': torch.sigmoid(self.opacities),  # Ensure opacities in [0, 1]
        }
    
    def get_covariance_matrix(self) -> torch.Tensor:
        """
        Compute the 3x3 covariance matrix for each Gaussian
        
        Returns:
            torch.Tensor: (N, 3, 3) covariance matrices
        """
        gaussians = self.forward()
        
        # Extract parameters
        positions = gaussians['positions']  # (N, 3)
        scales = gaussians['scales']  # (N, 3)
        rotations = gaussians['rotations']  # (N, 4)
        
        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
        
        # Create scale matrix
        scale_matrices = torch.diag_embed(scales)  # (N, 3, 3)
        
        # Compute covariance: R * S * S^T * R^T
        covariance = rotation_matrices @ scale_matrices @ scale_matrices.transpose(-2, -1) @ rotation_matrices.transpose(-2, -1)
        
        return covariance
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices
        
        Args:
            quaternions: (N, 4) quaternions [w, x, y, z]
            
        Returns:
            torch.Tensor: (N, 3, 3) rotation matrices
        """
        # Normalize quaternions
        quaternions = F.normalize(quaternions, dim=-1)
        
        w, x, y, z = quaternions.unbind(-1)
        
        # Rotation matrix from quaternion
        R = torch.stack([
            1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y,
            2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x,
            2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y
        ], dim=-1).view(-1, 3, 3)
        
        return R
    
    def prune_gaussians(self, threshold: float = 0.01):
        """
        Prune Gaussians with low opacity
        
        Args:
            threshold: Opacity threshold below which Gaussians are pruned
        """
        with torch.no_grad():
            opacities = torch.sigmoid(self.opacities).squeeze(-1)
            mask = opacities > threshold
            
            if mask.sum() < self.num_gaussians:
                # Update parameters to keep only high-opacity Gaussians
                self.positions = nn.Parameter(self.positions[mask])
                self.scales = nn.Parameter(self.scales[mask])
                self.rotations = nn.Parameter(self.rotations[mask])
                self.colors = nn.Parameter(self.colors[mask])
                self.opacities = nn.Parameter(self.opacities[mask])
                
                self.num_gaussians = mask.sum().item()
                print(f"Pruned {self.num_gaussians} Gaussians (threshold: {threshold})")
    
    def add_gaussians(self, num_new: int):
        """
        Add new Gaussians to the model
        
        Args:
            num_new: Number of new Gaussians to add
        """
        # Create new parameters
        new_positions = torch.rand(num_new, 3, device=self.device) * 2 - 1
        new_scales = torch.ones(num_new, 3, device=self.device) * 0.01
        new_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_new, 1)
        new_colors = torch.rand(num_new, 3, device=self.device)
        new_opacities = torch.ones(num_new, 1, device=self.device) * 0.5
        
        # Concatenate with existing parameters
        self.positions = nn.Parameter(torch.cat([self.positions, new_positions], dim=0))
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales], dim=0))
        self.rotations = nn.Parameter(torch.cat([self.rotations, new_rotations], dim=0))
        self.colors = nn.Parameter(torch.cat([self.colors, new_colors], dim=0))
        self.opacities = nn.Parameter(torch.cat([self.opacities, new_opacities], dim=0))
        
        self.num_gaussians += num_new
        print(f"Added {num_new} new Gaussians (total: {self.num_gaussians})") 