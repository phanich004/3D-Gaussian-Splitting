"""
Visualization utilities for 3D Gaussian Splatting
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from typing import Optional, Dict, List, Tuple
import os


def visualize_gaussians(gaussians: Dict[str, torch.Tensor], 
                       output_path: Optional[str] = None,
                       max_points: int = 10000) -> None:
    """
    Visualize 3D Gaussians as a scatter plot
    
    Args:
        gaussians: Dictionary containing Gaussian parameters
        output_path: Optional path to save visualization
        max_points: Maximum number of points to visualize
    """
    positions = gaussians['positions'].detach().cpu().numpy()
    colors = gaussians['colors'].detach().cpu().numpy()
    opacities = gaussians['opacities'].detach().cpu().numpy()
    scales = gaussians['scales'].detach().cpu().numpy()
    
    # Subsample if too many points
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        scales = scales[indices]
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by opacity
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=opacities.squeeze(), cmap='viridis', alpha=0.6, s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussian Visualization')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Opacity')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()


def save_rendered_images(rendered_images: torch.Tensor, output_path: str) -> None:
    """
    Save rendered images to file
    
    Args:
        rendered_images: (B, H, W, 3) or (B, 3, H, W) Rendered images
        output_path: Path to save the images
    """
    # Ensure images are in the correct format
    if rendered_images.shape[-1] == 3:
        # (B, H, W, 3) -> (B, 3, H, W)
        rendered_images = rendered_images.permute(0, 3, 1, 2)
    
    # Convert to numpy and denormalize
    images_np = rendered_images.detach().cpu().numpy()
    images_np = np.clip(images_np * 255, 0, 255).astype(np.uint8)
    
    # Save each image
    if len(images_np) == 1:
        # Single image
        image = images_np[0].transpose(1, 2, 0)
        Image.fromarray(image).save(output_path)
    else:
        # Multiple images - save as a grid
        grid_image = create_image_grid(images_np)
        Image.fromarray(grid_image).save(output_path)
    
    print(f"Saved rendered images to {output_path}")


def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid of images
    
    Args:
        images: (N, C, H, W) Images to arrange in grid
        grid_size: (rows, cols) Grid size. If None, auto-calculate
        
    Returns:
        Grid image as numpy array
    """
    N, C, H, W = images.shape
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
    else:
        rows, cols = grid_size
    
    # Create grid
    grid = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    
    for i in range(N):
        row = i // cols
        col = i % cols
        grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = images[i].transpose(1, 2, 0)
    
    return grid


def visualize_camera_poses(poses: Dict[str, torch.Tensor], 
                         output_path: Optional[str] = None) -> None:
    """
    Visualize camera poses as coordinate frames
    
    Args:
        poses: Dictionary containing camera poses
        output_path: Optional path to save visualization
    """
    positions = poses['positions'].detach().cpu().numpy()
    rotations = poses['rotations'].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=50, alpha=0.7)
    
    # Plot coordinate frames for each camera
    for i in range(min(len(positions), 20)):  # Limit to 20 cameras for clarity
        pos = positions[i]
        rot = rotations[i]
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(rot)
        
        # Camera coordinate frame axes (X, Y, Z)
        axes_length = 0.1
        for j, color in enumerate(['r', 'g', 'b']):
            axis = R[:, j] * axes_length
            ax.quiver(pos[0], pos[1], pos[2], 
                     axis[0], axis[1], axis[2], 
                     color=color, alpha=0.8, length=axes_length)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses Visualization')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved camera poses visualization to {output_path}")
    
    plt.show()


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    
    Args:
        quaternion: (4,) Quaternion [w, x, y, z]
        
    Returns:
        (3, 3) Rotation matrix
    """
    w, x, y, z = quaternion
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def create_comparison_image(rendered_images: torch.Tensor, target_images: torch.Tensor,
                          output_path: str) -> None:
    """
    Create a side-by-side comparison of rendered and target images
    
    Args:
        rendered_images: (B, 3, H, W) Rendered images
        target_images: (B, 3, H, W) Target images
        output_path: Path to save comparison
    """
    # Convert to numpy
    rendered_np = rendered_images.detach().cpu().numpy()
    target_np = target_images.detach().cpu().numpy()
    
    # Denormalize and clip
    rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
    target_np = np.clip(target_np * 255, 0, 255).astype(np.uint8)
    
    # Create comparison grid
    B, C, H, W = rendered_np.shape
    comparison = np.zeros((B * H, 2 * W, C), dtype=np.uint8)
    
    for i in range(B):
        # Add rendered image (left)
        comparison[i * H:(i + 1) * H, :W] = rendered_np[i].transpose(1, 2, 0)
        # Add target image (right)
        comparison[i * H:(i + 1) * H, W:2 * W] = target_np[i].transpose(1, 2, 0)
    
    # Save comparison
    Image.fromarray(comparison).save(output_path)
    print(f"Saved comparison to {output_path}")


def visualize_loss_curves(losses: Dict[str, List[float]], 
                         output_path: Optional[str] = None) -> None:
    """
    Visualize training loss curves
    
    Args:
        losses: Dictionary of loss names to loss values
        output_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 8))
    
    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name, alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curves to {output_path}")
    
    plt.show()


def create_video_from_images(image_paths: List[str], output_path: str, 
                           fps: int = 30) -> None:
    """
    Create a video from a sequence of images
    
    Args:
        image_paths: List of image paths
        output_path: Output video path
        fps: Frames per second
    """
    if not image_paths:
        print("No images provided")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for image_path in image_paths:
        if os.path.exists(image_path):
            frame = cv2.imread(image_path)
            video.write(frame)
    
    video.release()
    print(f"Saved video to {output_path}")


def visualize_gaussian_ellipsoids(gaussians: Dict[str, torch.Tensor], 
                                output_path: Optional[str] = None,
                                max_gaussians: int = 1000) -> None:
    """
    Visualize 3D Gaussians as ellipsoids
    
    Args:
        gaussians: Dictionary containing Gaussian parameters
        output_path: Optional path to save visualization
        max_gaussians: Maximum number of Gaussians to visualize
    """
    positions = gaussians['positions'].detach().cpu().numpy()
    scales = gaussians['scales'].detach().cpu().numpy()
    rotations = gaussians['rotations'].detach().cpu().numpy()
    opacities = gaussians['opacities'].detach().cpu().numpy()
    
    # Subsample if too many Gaussians
    if len(positions) > max_gaussians:
        indices = np.random.choice(len(positions), max_gaussians, replace=False)
        positions = positions[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opacities = opacities[indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ellipsoids
    for i in range(len(positions)):
        pos = positions[i]
        scale = scales[i]
        rot = rotations[i]
        opacity = opacities[i]
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(rot)
        
        # Create ellipsoid
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = scale[0] * np.outer(np.cos(u), np.sin(v))
        y = scale[1] * np.outer(np.sin(u), np.sin(v))
        z = scale[2] * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Apply rotation and translation
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                point = np.array([x[j, k], y[j, k], z[j, k]])
                rotated_point = R @ point
                x[j, k] = rotated_point[0] + pos[0]
                y[j, k] = rotated_point[1] + pos[1]
                z[j, k] = rotated_point[2] + pos[2]
        
        # Plot ellipsoid
        ax.plot_surface(x, y, z, alpha=opacity * 0.3, color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussian Ellipsoids')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ellipsoid visualization to {output_path}")
    
    plt.show() 