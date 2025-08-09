"""
Camera Pose Estimator for Pose-Free 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PoseEstimator(nn.Module):
    """
    Neural network for estimating camera poses from sparse views.
    This is a key innovation of the "No Pose at All" paper - learning
    camera poses jointly with the 3D Gaussian representation.
    """
    
    def __init__(self, num_cameras: int = 100, device: torch.device = torch.device('cuda')):
        super().__init__()
        self.num_cameras = num_cameras
        self.device = device
        
        # Initialize learnable camera poses
        self._init_camera_poses()
    
    def _init_camera_poses(self):
        """Initialize camera poses as learnable parameters"""
        # Camera positions: (N, 3) - initialized randomly
        self.camera_positions = nn.Parameter(
            torch.rand(self.num_cameras, 3, device=self.device) * 2 - 1
        )
        
        # Camera rotations: (N, 4) - quaternions, initialized as identity rotations
        self.camera_rotations = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_cameras, 1)
        )
        
        # Camera intrinsics: (N, 4) - fx, fy, cx, cy
        # Initialize with reasonable default values
        default_fx = 1000.0  # focal length x
        default_fy = 1000.0  # focal length y
        default_cx = 512.0   # principal point x
        default_cy = 512.0   # principal point y
        
        self.camera_intrinsics = nn.Parameter(
            torch.tensor([default_fx, default_fy, default_cx, default_cy], 
                        device=self.device).repeat(self.num_cameras, 1)
        )
    
    def forward(self, camera_ids: torch.Tensor) -> dict:
        """
        Forward pass to get camera poses and intrinsics
        
        Args:
            camera_ids: (B,) camera IDs for the batch
            
        Returns:
            dict: Dictionary containing:
                - positions: (B, 3) camera positions
                - rotations: (B, 4) camera rotations (quaternions)
                - intrinsics: (B, 4) camera intrinsics [fx, fy, cx, cy]
                - view_matrices: (B, 4, 4) view matrices
                - projection_matrices: (B, 4, 4) projection matrices
        """
        # Get camera parameters for the batch
        positions = self.camera_positions[camera_ids]  # (B, 3)
        rotations = F.normalize(self.camera_rotations[camera_ids], dim=-1)  # (B, 4)
        intrinsics = self.camera_intrinsics[camera_ids]  # (B, 4)
        
        # Convert quaternions to rotation matrices
        rotation_matrices = self._quaternion_to_rotation_matrix(rotations)  # (B, 3, 3)
        
        # Build view matrices: [R | -R*t]
        translation = -rotation_matrices @ positions.unsqueeze(-1)  # (B, 3, 1)
        view_matrix = torch.cat([rotation_matrices, translation], dim=-1)  # (B, 3, 4)
        
        # Add homogeneous coordinate row
        homogeneous_row = torch.tensor([0, 0, 0, 1], device=self.device).repeat(view_matrix.shape[0], 1, 1)
        view_matrix = torch.cat([view_matrix, homogeneous_row], dim=1)  # (B, 4, 4)
        
        # Build projection matrices from intrinsics
        projection_matrix = self._build_projection_matrix(intrinsics)  # (B, 4, 4)
        
        return {
            'positions': positions,
            'rotations': rotations,
            'intrinsics': intrinsics,
            'view_matrices': view_matrix,
            'projection_matrices': projection_matrix,
            'rotation_matrices': rotation_matrices
        }
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices
        
        Args:
            quaternions: (B, 4) quaternions [w, x, y, z]
            
        Returns:
            torch.Tensor: (B, 3, 3) rotation matrices
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
    
    def _build_projection_matrix(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Build projection matrix from camera intrinsics
        
        Args:
            intrinsics: (B, 4) [fx, fy, cx, cy]
            
        Returns:
            torch.Tensor: (B, 4, 4) projection matrices
        """
        fx, fy, cx, cy = intrinsics.unbind(-1)
        batch_size = intrinsics.shape[0]
        
        # Create perspective projection matrix
        # Assuming near=0.1, far=100.0, image_size=1024
        near, far = 0.1, 100.0
        image_size = 1024
        
        projection_matrix = torch.zeros(batch_size, 4, 4, device=self.device)
        
        # Perspective projection matrix
        projection_matrix[:, 0, 0] = 2 * fx / image_size
        projection_matrix[:, 1, 1] = 2 * fy / image_size
        projection_matrix[:, 0, 2] = -2 * cx / image_size + 1
        projection_matrix[:, 1, 2] = -2 * cy / image_size + 1
        projection_matrix[:, 2, 2] = -(far + near) / (far - near)
        projection_matrix[:, 2, 3] = -2 * far * near / (far - near)
        projection_matrix[:, 3, 2] = -1
        
        return projection_matrix
    
    def get_camera_to_world_matrix(self, camera_ids: torch.Tensor) -> torch.Tensor:
        """
        Get camera-to-world transformation matrix
        
        Args:
            camera_ids: (B,) camera IDs
            
        Returns:
            torch.Tensor: (B, 4, 4) camera-to-world matrices
        """
        poses = self.forward(camera_ids)
        view_matrix = poses['view_matrices']
        
        # Invert view matrix to get camera-to-world matrix
        # For a view matrix [R | t], the inverse is [R^T | -R^T * t]
        rotation = view_matrix[:, :3, :3]  # (B, 3, 3)
        translation = view_matrix[:, :3, 3:]  # (B, 3, 1)
        
        # Camera-to-world transformation
        rotation_transpose = rotation.transpose(-2, -1)  # (B, 3, 3)
        translation_world = -rotation_transpose @ translation  # (B, 3, 1)
        
        camera_to_world = torch.cat([rotation_transpose, translation_world], dim=-1)  # (B, 3, 4)
        homogeneous_row = torch.tensor([0, 0, 0, 1], device=self.device).repeat(camera_to_world.shape[0], 1, 1)
        camera_to_world = torch.cat([camera_to_world, homogeneous_row], dim=1)  # (B, 4, 4)
        
        return camera_to_world
    
    def compute_relative_pose(self, camera_ids1: torch.Tensor, camera_ids2: torch.Tensor) -> dict:
        """
        Compute relative pose between two cameras
        
        Args:
            camera_ids1: (B,) first camera IDs
            camera_ids2: (B,) second camera IDs
            
        Returns:
            dict: Relative pose information
        """
        poses1 = self.forward(camera_ids1)
        poses2 = self.forward(camera_ids2)
        
        # Get camera-to-world matrices
        c2w1 = self.get_camera_to_world_matrix(camera_ids1)  # (B, 4, 4)
        c2w2 = self.get_camera_to_world_matrix(camera_ids2)  # (B, 4, 4)
        
        # Compute relative pose: c2w2 @ inv(c2w1)
        inv_c2w1 = torch.inverse(c2w1)  # (B, 4, 4)
        relative_pose = c2w2 @ inv_c2w1  # (B, 4, 4)
        
        # Extract rotation and translation
        relative_rotation = relative_pose[:, :3, :3]  # (B, 3, 3)
        relative_translation = relative_pose[:, :3, 3]  # (B, 3)
        
        return {
            'relative_rotation': relative_rotation,
            'relative_translation': relative_translation,
            'relative_pose_matrix': relative_pose
        }
    
    def regularize_poses(self) -> torch.Tensor:
        """
        Compute pose regularization loss to encourage smooth camera trajectories
        
        Returns:
            torch.Tensor: Regularization loss
        """
        # Sort cameras by ID to ensure temporal ordering
        sorted_indices = torch.arange(self.num_cameras, device=self.device)
        
        # Get consecutive camera poses
        poses_curr = self.forward(sorted_indices[:-1])
        poses_next = self.forward(sorted_indices[1:])
        
        # Compute pose differences
        pos_diff = poses_next['positions'] - poses_curr['positions']
        rot_diff = poses_next['rotations'] - poses_curr['rotations']
        
        # Regularization losses
        pos_smoothness = torch.mean(torch.norm(pos_diff, dim=-1) ** 2)
        rot_smoothness = torch.mean(torch.norm(rot_diff, dim=-1) ** 2)
        
        return pos_smoothness + rot_smoothness 