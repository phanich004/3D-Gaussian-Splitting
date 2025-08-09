"""
Geometric loss functions for 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class GeometricLoss(nn.Module):
    """
    Geometric loss functions that enforce geometric consistency between views.
    This includes depth consistency, normal consistency, and epipolar geometry constraints.
    """
    
    def __init__(self, loss_type: str = 'depth_consistency', weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
    
    def forward(self, gaussians: Dict[str, torch.Tensor], poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute geometric loss
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            poses: Dictionary of camera parameters
            
        Returns:
            torch.Tensor: Geometric loss
        """
        if self.loss_type == 'depth_consistency':
            return self._compute_depth_consistency_loss(gaussians, poses)
        elif self.loss_type == 'normal_consistency':
            return self._compute_normal_consistency_loss(gaussians, poses)
        elif self.loss_type == 'epipolar':
            return self._compute_epipolar_loss(gaussians, poses)
        elif self.loss_type == 'combined':
            depth_loss = self._compute_depth_consistency_loss(gaussians, poses)
            normal_loss = self._compute_normal_consistency_loss(gaussians, poses)
            epipolar_loss = self._compute_epipolar_loss(gaussians, poses)
            return depth_loss + normal_loss + epipolar_loss
        else:
            raise ValueError(f"Unknown geometric loss type: {self.loss_type}")
    
    def _compute_depth_consistency_loss(self, gaussians: Dict[str, torch.Tensor], 
                                      poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute depth consistency loss between different views
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            poses: Dictionary of camera parameters
            
        Returns:
            torch.Tensor: Depth consistency loss
        """
        positions = gaussians['positions']  # (N, 3)
        opacities = gaussians['opacities']  # (N, 1)
        
        # Get camera poses
        view_matrices = poses['view_matrices']  # (B, 4, 4)
        batch_size = view_matrices.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Compute depth maps for each view
        depth_maps = []
        for b in range(batch_size):
            view_matrix = view_matrices[b:b+1]  # (1, 4, 4)
            
            # Transform points to camera space
            homogeneous_positions = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)  # (N, 4)
            camera_positions = (view_matrix @ homogeneous_positions.unsqueeze(-1)).squeeze(-1)  # (N, 4)
            
            # Get depth values (Z coordinate)
            depths = camera_positions[:, 2]  # (N,)
            
            # Weight by opacity
            weighted_depths = depths * opacities.squeeze(-1)
            
            depth_maps.append(weighted_depths)
        
        # Compute depth consistency between consecutive views
        depth_consistency_loss = 0.0
        for i in range(len(depth_maps) - 1):
            depth_diff = torch.abs(depth_maps[i] - depth_maps[i + 1])
            depth_consistency_loss += torch.mean(depth_diff)
        
        return depth_consistency_loss / max(1, len(depth_maps) - 1)
    
    def _compute_normal_consistency_loss(self, gaussians: Dict[str, torch.Tensor], 
                                       poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute normal consistency loss
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            poses: Dictionary of camera parameters
            
        Returns:
            torch.Tensor: Normal consistency loss
        """
        positions = gaussians['positions']  # (N, 3)
        rotations = gaussians['rotations']  # (N, 4)
        opacities = gaussians['opacities']  # (N, 1)
        
        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
        
        # Extract normal vectors (assuming Z-axis as normal)
        normals = rotation_matrices[:, :, 2]  # (N, 3)
        
        # Compute normal consistency between nearby Gaussians
        normal_consistency_loss = 0.0
        
        # Use k-nearest neighbors to find nearby Gaussians
        k = min(10, len(positions) - 1)
        if k > 0:
            # Compute pairwise distances
            dist_matrix = torch.cdist(positions, positions)  # (N, N)
            
            # Find k nearest neighbors (excluding self)
            _, nearest_indices = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)  # (N, k+1)
            nearest_indices = nearest_indices[:, 1:]  # (N, k) - exclude self
            
            # Compute normal consistency with neighbors
            for i in range(len(positions)):
                neighbor_normals = normals[nearest_indices[i]]  # (k, 3)
                current_normal = normals[i:i+1]  # (1, 3)
                
                # Compute cosine similarity
                cosine_similarity = torch.sum(current_normal * neighbor_normals, dim=-1)  # (k,)
                
                # Normal consistency loss (1 - cosine_similarity)
                consistency_loss = torch.mean(1 - cosine_similarity)
                
                # Weight by opacity
                opacity_weight = opacities[i, 0]
                normal_consistency_loss += opacity_weight * consistency_loss
        
        return normal_consistency_loss / max(1, len(positions))
    
    def _compute_epipolar_loss(self, gaussians: Dict[str, torch.Tensor], 
                             poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute epipolar geometry loss to enforce multi-view consistency
        
        Args:
            gaussians: Dictionary of Gaussian parameters
            poses: Dictionary of camera parameters
            
        Returns:
            torch.Tensor: Epipolar loss
        """
        positions = gaussians['positions']  # (N, 3)
        opacities = gaussians['opacities']  # (N, 1)
        
        # Get camera poses
        view_matrices = poses['view_matrices']  # (B, 4, 4)
        batch_size = view_matrices.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=positions.device)
        
        epipolar_loss = 0.0
        num_pairs = 0
        
        # Compute epipolar loss for camera pairs
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Get camera matrices
                P1 = view_matrices[i]  # (4, 4)
                P2 = view_matrices[j]  # (4, 4)
                
                # Compute fundamental matrix
                F_matrix = self._compute_fundamental_matrix(P1, P2)
                
                # Project points to both views
                homogeneous_points = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)  # (N, 4)
                
                # Project to first view
                points1 = (P1 @ homogeneous_points.unsqueeze(-1)).squeeze(-1)  # (N, 4)
                points1 = points1[:, :2] / points1[:, 2:3].clamp(min=1e-8)  # (N, 2)
                
                # Project to second view
                points2 = (P2 @ homogeneous_points.unsqueeze(-1)).squeeze(-1)  # (N, 4)
                points2 = points2[:, :2] / points2[:, 2:3].clamp(min=1e-8)  # (N, 2)
                
                # Compute epipolar error
                epipolar_error = self._compute_epipolar_error(points1, points2, F_matrix)
                
                # Weight by opacity
                weighted_error = torch.sum(epipolar_error * opacities.squeeze(-1))
                
                epipolar_loss += weighted_error
                num_pairs += 1
        
        return epipolar_loss / max(1, num_pairs) if num_pairs > 0 else torch.tensor(0.0, device=positions.device)
    
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
    
    def _compute_fundamental_matrix(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        """
        Compute fundamental matrix from two camera matrices
        
        Args:
            P1: (4, 4) First camera matrix
            P2: (4, 4) Second camera matrix
            
        Returns:
            torch.Tensor: (3, 3) Fundamental matrix
        """
        # Extract camera centers
        C1 = -torch.inverse(P1[:3, :3]) @ P1[:3, 3]  # (3,)
        C2 = -torch.inverse(P2[:3, :3]) @ P2[:3, 3]  # (3,)
        
        # Compute translation vector
        t = C2 - C1  # (3,)
        
        # Compute essential matrix
        t_cross = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], device=t.device, dtype=t.dtype)
        
        # Rotation from P1 to P2
        R1 = P1[:3, :3]
        R2 = P2[:3, :3]
        R = R2 @ R1.T  # (3, 3)
        
        # Essential matrix
        E = t_cross @ R  # (3, 3)
        
        # Fundamental matrix (assuming identity intrinsics for simplicity)
        F_matrix = E  # (3, 3)
        
        return F_matrix
    
    def _compute_epipolar_error(self, points1: torch.Tensor, points2: torch.Tensor, 
                              F_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute epipolar error for corresponding points
        
        Args:
            points1: (N, 2) Points in first view
            points2: (N, 2) Points in second view
            F_matrix: (3, 3) Fundamental matrix
            
        Returns:
            torch.Tensor: (N,) Epipolar errors
        """
        # Convert to homogeneous coordinates
        points1_homo = torch.cat([points1, torch.ones_like(points1[:, :1])], dim=-1)  # (N, 3)
        points2_homo = torch.cat([points2, torch.ones_like(points2[:, :1])], dim=-1)  # (N, 3)
        
        # Compute epipolar lines
        epipolar_lines = (F_matrix @ points1_homo.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        
        # Compute epipolar error: (x2^T * F * x1) / sqrt(a^2 + b^2)
        errors = torch.abs(torch.sum(points2_homo * epipolar_lines, dim=-1))  # (N,)
        line_norms = torch.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2)  # (N,)
        
        # Normalize by line norm
        epipolar_error = errors / (line_norms + 1e-8)  # (N,)
        
        return epipolar_error


class DepthConsistencyLoss(nn.Module):
    """
    Specialized depth consistency loss for 3D Gaussian Splatting
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, rendered_depths: torch.Tensor, target_depths: Optional[torch.Tensor] = None,
               gaussians: Optional[Dict[str, torch.Tensor]] = None,
               poses: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute depth consistency loss
        
        Args:
            rendered_depths: (B, H, W) Rendered depth maps
            target_depths: (B, H, W) Target depth maps (optional)
            gaussians: Dictionary of Gaussian parameters (optional)
            poses: Dictionary of camera parameters (optional)
            
        Returns:
            torch.Tensor: Depth consistency loss
        """
        if target_depths is not None:
            # Direct depth comparison
            depth_loss = F.mse_loss(rendered_depths, target_depths)
        else:
            # Depth consistency between views
            depth_loss = self._compute_multi_view_depth_consistency(rendered_depths)
        
        return self.weight * depth_loss
    
    def _compute_multi_view_depth_consistency(self, rendered_depths: torch.Tensor) -> torch.Tensor:
        """
        Compute depth consistency between multiple views
        
        Args:
            rendered_depths: (B, H, W) Rendered depth maps
            
        Returns:
            torch.Tensor: Depth consistency loss
        """
        if rendered_depths.shape[0] < 2:
            return torch.tensor(0.0, device=rendered_depths.device)
        
        # Compute depth differences between consecutive views
        depth_diffs = []
        for i in range(rendered_depths.shape[0] - 1):
            diff = torch.abs(rendered_depths[i] - rendered_depths[i+1])
            depth_diffs.append(torch.mean(diff))
        
        return torch.mean(torch.stack(depth_diffs)) 