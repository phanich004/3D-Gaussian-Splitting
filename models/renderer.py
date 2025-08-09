"""
Differentiable 3D Gaussian Splatting Renderer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class GaussianRenderer(nn.Module):
    """
    Differentiable 3D Gaussian Splatting renderer.
    This implements the core rendering algorithm for 3D Gaussian splatting.
    """
    
    def __init__(self, image_size: int = 1024, device: torch.device = torch.device('cuda')):
        super().__init__()
        self.image_size = image_size
        self.device = device
        
        # Create coordinate grid for rasterization
        self._create_coordinate_grid()
    
    def _create_coordinate_grid(self):
        """Create coordinate grid for rasterization"""
        # Create pixel coordinates
        x_coords = torch.arange(self.image_size, device=self.device, dtype=torch.float32)
        y_coords = torch.arange(self.image_size, device=self.device, dtype=torch.float32)
        
        # Create meshgrid
        self.grid_x, self.grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Reshape to (H*W, 2)
        self.pixel_coords = torch.stack([self.grid_x.flatten(), self.grid_y.flatten()], dim=-1)
    
    def forward(self, gaussians: Dict[str, torch.Tensor], poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for rendering images from 3D Gaussians
        
        Args:
            gaussians: Dictionary containing Gaussian parameters:
                - positions: (N, 3) Gaussian positions
                - scales: (N, 3) Gaussian scales
                - rotations: (N, 4) Gaussian rotations (quaternions)
                - colors: (N, 3) Gaussian colors (RGB)
                - opacities: (N, 1) Gaussian opacities
            poses: Dictionary containing camera parameters:
                - view_matrices: (B, 4, 4) View matrices
                - projection_matrices: (B, 4, 4) Projection matrices
                
        Returns:
            torch.Tensor: (B, H, W, 3) Rendered images
        """
        batch_size = poses['view_matrices'].shape[0]
        rendered_images = []
        
        for b in range(batch_size):
            # Get camera parameters for this batch
            view_matrix = poses['view_matrices'][b:b+1]  # (1, 4, 4)
            projection_matrix = poses['projection_matrices'][b:b+1]  # (1, 4, 4)
            
            # Render single image
            rendered_image = self._render_single_image(gaussians, view_matrix, projection_matrix)
            rendered_images.append(rendered_image)
        
        # Stack batch
        rendered_images = torch.stack(rendered_images, dim=0)  # (B, H, W, 3)
        return rendered_images
    
    def _render_single_image(self, gaussians: Dict[str, torch.Tensor], 
                           view_matrix: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        Render a single image from 3D Gaussians
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            view_matrix: (1, 4, 4) View matrix
            projection_matrix: (1, 4, 4) Projection matrix
            
        Returns:
            torch.Tensor: (H, W, 3) Rendered image
        """
        # Extract Gaussian parameters
        positions = gaussians['positions']  # (N, 3)
        scales = gaussians['scales']  # (N, 3)
        rotations = gaussians['rotations']  # (N, 4)
        colors = gaussians['colors']  # (N, 3)
        opacities = gaussians['opacities']  # (N, 1)
        
        # Transform Gaussians to camera space
        camera_positions, camera_scales, camera_rotations = self._transform_gaussians(
            positions, scales, rotations, view_matrix
        )
        
        # Project Gaussians to 2D
        screen_positions, screen_scales = self._project_gaussians(
            camera_positions, camera_scales, projection_matrix
        )
        
        # Sort Gaussians by depth for back-to-front rendering
        sorted_indices = torch.argsort(camera_positions[:, 2], descending=True)
        
        # Initialize output image
        rendered_image = torch.zeros(self.image_size, self.image_size, 3, device=self.device)
        rendered_alpha = torch.zeros(self.image_size, self.image_size, 1, device=self.device)
        
        # Render Gaussians back-to-front
        for idx in sorted_indices:
            # Get Gaussian parameters
            pos_2d = screen_positions[idx:idx+1]  # (1, 2)
            scale_2d = screen_scales[idx:idx+1]  # (1, 2)
            color = colors[idx:idx+1]  # (1, 3)
            opacity = opacities[idx:idx+1]  # (1, 1)
            
            # Render this Gaussian
            gaussian_image, gaussian_alpha = self._render_gaussian(
                pos_2d, scale_2d, color, opacity
            )
            
            # Alpha compositing
            alpha = gaussian_alpha * (1 - rendered_alpha)
            rendered_image = rendered_image + alpha * gaussian_image
            rendered_alpha = rendered_alpha + alpha
        
        return rendered_image
    
    def _transform_gaussians(self, positions: torch.Tensor, scales: torch.Tensor, 
                           rotations: torch.Tensor, view_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform Gaussians from world space to camera space
        
        Args:
            positions: (N, 3) Gaussian positions in world space
            scales: (N, 3) Gaussian scales in world space
            rotations: (N, 4) Gaussian rotations in world space
            view_matrix: (1, 4, 4) View matrix
            
        Returns:
            Tuple of (camera_positions, camera_scales, camera_rotations)
        """
        # Transform positions
        homogeneous_positions = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)  # (N, 4)
        camera_positions = (view_matrix @ homogeneous_positions.unsqueeze(-1)).squeeze(-1)[:, :3]  # (N, 3)
        
        # Transform scales (approximate)
        camera_scales = scales  # Simplified for now
        
        # Transform rotations (approximate)
        camera_rotations = rotations  # Simplified for now
        
        return camera_positions, camera_scales, camera_rotations
    
    def _project_gaussians(self, camera_positions: torch.Tensor, camera_scales: torch.Tensor,
                          projection_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D Gaussians to 2D screen space
        
        Args:
            camera_positions: (N, 3) Gaussian positions in camera space
            camera_scales: (N, 3) Gaussian scales in camera space
            projection_matrix: (1, 4, 4) Projection matrix
            
        Returns:
            Tuple of (screen_positions, screen_scales)
        """
        # Project positions
        homogeneous_positions = torch.cat([camera_positions, torch.ones_like(camera_positions[:, :1])], dim=-1)  # (N, 4)
        projected_positions = (projection_matrix @ homogeneous_positions.unsqueeze(-1)).squeeze(-1)  # (N, 4)
        
        # Perspective divide
        projected_positions = projected_positions[:, :3] / projected_positions[:, 3:4].clamp(min=1e-8)
        
        # Convert to screen coordinates
        screen_positions = torch.stack([
            (projected_positions[:, 0] + 1) * self.image_size / 2,
            (projected_positions[:, 1] + 1) * self.image_size / 2
        ], dim=-1)  # (N, 2)
        
        # Project scales (simplified)
        screen_scales = camera_scales[:, :2] * self.image_size  # (N, 2)
        
        return screen_positions, screen_scales
    
    def _render_gaussian(self, pos_2d: torch.Tensor, scale_2d: torch.Tensor, 
                        color: torch.Tensor, opacity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a single 2D Gaussian
        
        Args:
            pos_2d: (1, 2) 2D position
            scale_2d: (1, 2) 2D scale
            color: (1, 3) Color
            opacity: (1, 1) Opacity
            
        Returns:
            Tuple of (gaussian_image, gaussian_alpha)
        """
        # Create coordinate grid for this Gaussian
        x_center, y_center = pos_2d[0, 0], pos_2d[0, 1]
        x_scale, y_scale = scale_2d[0, 0], scale_2d[0, 1]
        
        # Define rendering region
        x_min = max(0, int(x_center - 3 * x_scale))
        x_max = min(self.image_size, int(x_center + 3 * x_scale))
        y_min = max(0, int(y_center - 3 * y_scale))
        y_max = min(self.image_size, int(y_center + 3 * y_scale))
        
        if x_max <= x_min or y_max <= y_min:
            # Gaussian outside image bounds
            return torch.zeros(self.image_size, self.image_size, 3, device=self.device), \
                   torch.zeros(self.image_size, self.image_size, 1, device=self.device)
        
        # Create local coordinate grid
        x_coords = torch.arange(x_min, x_max, device=self.device, dtype=torch.float32)
        y_coords = torch.arange(y_min, y_max, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Compute Gaussian values
        dx = (grid_x - x_center) / x_scale
        dy = (grid_y - y_center) / y_scale
        
        # 2D Gaussian function
        gaussian_values = torch.exp(-0.5 * (dx**2 + dy**2))  # (H, W)
        gaussian_values = gaussian_values * opacity[0, 0]
        
        # Create output tensors
        gaussian_image = torch.zeros(self.image_size, self.image_size, 3, device=self.device)
        gaussian_alpha = torch.zeros(self.image_size, self.image_size, 1, device=self.device)
        
        # Fill the region
        gaussian_image[y_min:y_max, x_min:x_max] = gaussian_values.unsqueeze(-1) * color[0]
        gaussian_alpha[y_min:y_max, x_min:x_max] = gaussian_values.unsqueeze(-1)
        
        return gaussian_image, gaussian_alpha
    
    def compute_depth_map(self, gaussians: Dict[str, torch.Tensor], 
                         poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute depth map from 3D Gaussians
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            poses: Dictionary of camera parameters
            
        Returns:
            torch.Tensor: (B, H, W) Depth maps
        """
        # Similar to forward pass but compute depth instead of color
        batch_size = poses['view_matrices'].shape[0]
        depth_maps = []
        
        for b in range(batch_size):
            view_matrix = poses['view_matrices'][b:b+1]
            projection_matrix = poses['projection_matrices'][b:b+1]
            
            # Get camera-space positions
            positions = gaussians['positions']
            homogeneous_positions = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)
            camera_positions = (view_matrix @ homogeneous_positions.unsqueeze(-1)).squeeze(-1)[:, 2]  # Z-coordinate
            
            # Project to screen space and render depth
            depth_map = self._render_depth(gaussians, camera_positions, view_matrix, projection_matrix)
            depth_maps.append(depth_map)
        
        return torch.stack(depth_maps, dim=0)
    
    def _render_depth(self, gaussians: Dict[str, torch.Tensor], camera_depths: torch.Tensor,
                     view_matrix: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Render depth map from Gaussians"""
        # Simplified depth rendering - similar to color rendering but using depth values
        depth_map = torch.zeros(self.image_size, self.image_size, device=self.device)
        weight_map = torch.zeros(self.image_size, self.image_size, device=self.device)
        
        # Project Gaussians and accumulate depths
        positions = gaussians['positions']
        opacities = gaussians['opacities']
        
        for i in range(len(positions)):
            # Project this Gaussian
            pos_2d, _ = self._project_gaussians(
                positions[i:i+1], 
                gaussians['scales'][i:i+1], 
                projection_matrix
            )
            
            # Render Gaussian contribution to depth
            x_center, y_center = pos_2d[0, 0], pos_2d[0, 1]
            x_min = max(0, int(x_center - 10))
            x_max = min(self.image_size, int(x_center + 10))
            y_min = max(0, int(y_center - 10))
            y_max = min(self.image_size, int(y_center + 10))
            
            if x_max > x_min and y_max > y_min:
                weight = opacities[i, 0]
                depth_map[y_min:y_max, x_min:x_max] += weight * camera_depths[i]
                weight_map[y_min:y_max, x_min:x_max] += weight
        
        # Normalize by weights
        depth_map = depth_map / (weight_map + 1e-8)
        return depth_map 