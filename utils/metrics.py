"""
Evaluation metrics for 3D Gaussian Splatting
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
import math


def compute_psnr(rendered_images: torch.Tensor, target_images: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        rendered_images: (B, C, H, W) or (B, H, W, C) Rendered images
        target_images: (B, C, H, W) or (B, H, W, C) Target images
        
    Returns:
        PSNR value
    """
    # Ensure images are in the correct format
    if rendered_images.shape[-1] == 3:
        rendered_images = rendered_images.permute(0, 3, 1, 2)
    if target_images.shape[-1] == 3:
        target_images = target_images.permute(0, 3, 1, 2)
    
    # Normalize to [0, 1] if needed
    if rendered_images.max() > 1.0:
        rendered_images = rendered_images / 255.0
    if target_images.max() > 1.0:
        target_images = target_images / 255.0
    
    # Compute MSE
    mse = F.mse_loss(rendered_images, target_images)
    
    # PSNR = 20 * log10(1 / sqrt(MSE))
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr


def compute_ssim(rendered_images: torch.Tensor, target_images: torch.Tensor, 
                window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        rendered_images: (B, C, H, W) or (B, H, W, C) Rendered images
        target_images: (B, C, H, W) or (B, H, W, C) Target images
        window_size: Window size for SSIM computation
        
    Returns:
        SSIM value
    """
    # Ensure images are in the correct format
    if rendered_images.shape[-1] == 3:
        rendered_images = rendered_images.permute(0, 3, 1, 2)
    if target_images.shape[-1] == 3:
        target_images = target_images.permute(0, 3, 1, 2)
    
    # Normalize to [0, 1] if needed
    if rendered_images.max() > 1.0:
        rendered_images = rendered_images / 255.0
    if target_images.max() > 1.0:
        target_images = target_images / 255.0
    
    # Compute SSIM for each channel
    ssim_values = []
    for i in range(rendered_images.shape[1]):  # For each channel
        ssim_chan = _compute_ssim_channel(rendered_images[:, i:i+1], target_images[:, i:i+1], window_size)
        ssim_values.append(ssim_chan)
    
    # Average across channels
    return torch.mean(torch.stack(ssim_values)).item()


def _compute_ssim_channel(rendered_channel: torch.Tensor, target_channel: torch.Tensor, 
                         window_size: int) -> torch.Tensor:
    """
    Compute SSIM for a single channel
    
    Args:
        rendered_channel: (B, 1, H, W) Rendered channel
        target_channel: (B, 1, H, W) Target channel
        window_size: Window size
        
    Returns:
        SSIM value for the channel
    """
    # SSIM parameters
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute means using average pooling
    mu_x = F.avg_pool2d(rendered_channel, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(target_channel, window_size, stride=1, padding=window_size//2)
    
    # Compute variances and covariance
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.avg_pool2d(rendered_channel ** 2, window_size, stride=1, padding=window_size//2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target_channel ** 2, window_size, stride=1, padding=window_size//2) - mu_y_sq
    sigma_xy = F.avg_pool2d(rendered_channel * target_channel, window_size, stride=1, padding=window_size//2) - mu_xy
    
    # SSIM formula
    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return torch.mean(ssim)


def compute_lpips(rendered_images: torch.Tensor, target_images: torch.Tensor) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity)
    
    Args:
        rendered_images: (B, C, H, W) or (B, H, W, C) Rendered images
        target_images: (B, C, H, W) or (B, H, W, C) Target images
        
    Returns:
        LPIPS value
    """
    try:
        import lpips
    except ImportError:
        print("Warning: LPIPS not available. Returning 0.0")
        return 0.0
    
    # Ensure images are in the correct format
    if rendered_images.shape[-1] == 3:
        rendered_images = rendered_images.permute(0, 3, 1, 2)
    if target_images.shape[-1] == 3:
        target_images = target_images.permute(0, 3, 1, 2)
    
    # Normalize to [-1, 1] for LPIPS
    rendered_normalized = 2 * rendered_images - 1
    target_normalized = 2 * target_images - 1
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='vgg')
    
    # Compute LPIPS
    lpips_value = lpips_fn(rendered_normalized, target_normalized)
    return torch.mean(lpips_value).item()


def compute_depth_accuracy(rendered_depths: torch.Tensor, target_depths: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute depth accuracy metrics
    
    Args:
        rendered_depths: (B, H, W) Rendered depth maps
        target_depths: (B, H, W) Target depth maps
        mask: (B, H, W) Optional mask for valid pixels
        
    Returns:
        Dictionary of depth metrics
    """
    if mask is not None:
        rendered_depths = rendered_depths[mask]
        target_depths = target_depths[mask]
    
    # Compute metrics
    abs_error = torch.abs(rendered_depths - target_depths)
    rel_error = abs_error / (target_depths + 1e-8)
    
    metrics = {
        'depth_abs_error': torch.mean(abs_error).item(),
        'depth_rel_error': torch.mean(rel_error).item(),
        'depth_rmse': torch.sqrt(torch.mean((rendered_depths - target_depths) ** 2)).item(),
    }
    
    # Compute accuracy thresholds
    thresholds = [1.25, 1.25**2, 1.25**3]
    for threshold in thresholds:
        accuracy = torch.mean((torch.maximum(rendered_depths / target_depths, target_depths / rendered_depths) < threshold).float())
        metrics[f'depth_acc_{threshold}'] = accuracy.item()
    
    return metrics


def compute_pose_accuracy(estimated_poses: torch.Tensor, ground_truth_poses: torch.Tensor) -> Dict[str, float]:
    """
    Compute pose accuracy metrics
    
    Args:
        estimated_poses: (B, 4, 4) Estimated camera poses
        ground_truth_poses: (B, 4, 4) Ground truth camera poses
        
    Returns:
        Dictionary of pose metrics
    """
    # Extract rotation and translation
    estimated_rotations = estimated_poses[:, :3, :3]  # (B, 3, 3)
    estimated_translations = estimated_poses[:, :3, 3]  # (B, 3)
    
    gt_rotations = ground_truth_poses[:, :3, :3]  # (B, 3, 3)
    gt_translations = ground_truth_poses[:, :3, 3]  # (B, 3)
    
    # Rotation error (in degrees)
    rotation_error = []
    for i in range(len(estimated_rotations)):
        R_error = torch.matmul(estimated_rotations[i], gt_rotations[i].T)
        trace = torch.trace(R_error)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        rotation_error.append(torch.rad2deg(angle))
    
    rotation_error = torch.tensor(rotation_error)
    
    # Translation error
    translation_error = torch.norm(estimated_translations - gt_translations, dim=-1)
    
    metrics = {
        'rotation_error_mean': torch.mean(rotation_error).item(),
        'rotation_error_std': torch.std(rotation_error).item(),
        'translation_error_mean': torch.mean(translation_error).item(),
        'translation_error_std': torch.std(translation_error).item(),
    }
    
    return metrics


def compute_gaussian_quality_metrics(gaussians: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute quality metrics for 3D Gaussians
    
    Args:
        gaussians: Dictionary containing Gaussian parameters
        
    Returns:
        Dictionary of quality metrics
    """
    positions = gaussians['positions']  # (N, 3)
    scales = gaussians['scales']  # (N, 3)
    opacities = gaussians['opacities']  # (N, 1)
    colors = gaussians['colors']  # (N, 3)
    
    # Number of Gaussians
    num_gaussians = len(positions)
    
    # Opacity distribution
    opacity_mean = torch.mean(opacities).item()
    opacity_std = torch.std(opacities).item()
    
    # Scale distribution
    scale_mean = torch.mean(scales).item()
    scale_std = torch.std(scales).item()
    
    # Position distribution
    position_range = torch.max(positions) - torch.min(positions)
    
    # Color distribution
    color_mean = torch.mean(colors).item()
    color_std = torch.std(colors).item()
    
    # Spatial density (Gaussians per unit volume)
    volume = position_range ** 3
    spatial_density = num_gaussians / volume if volume > 0 else 0
    
    metrics = {
        'num_gaussians': num_gaussians,
        'opacity_mean': opacity_mean,
        'opacity_std': opacity_std,
        'scale_mean': scale_mean,
        'scale_std': scale_std,
        'position_range': position_range.item(),
        'color_mean': color_mean,
        'color_std': color_std,
        'spatial_density': spatial_density.item(),
    }
    
    return metrics


def compute_reconstruction_metrics(rendered_images: torch.Tensor, target_images: torch.Tensor,
                                 rendered_depths: Optional[torch.Tensor] = None,
                                 target_depths: Optional[torch.Tensor] = None,
                                 estimated_poses: Optional[torch.Tensor] = None,
                                 ground_truth_poses: Optional[torch.Tensor] = None,
                                 gaussians: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics
    
    Args:
        rendered_images: Rendered images
        target_images: Target images
        rendered_depths: Optional rendered depth maps
        target_depths: Optional target depth maps
        estimated_poses: Optional estimated camera poses
        ground_truth_poses: Optional ground truth camera poses
        gaussians: Optional Gaussian parameters
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Image quality metrics
    metrics['psnr'] = compute_psnr(rendered_images, target_images)
    metrics['ssim'] = compute_ssim(rendered_images, target_images)
    metrics['lpips'] = compute_lpips(rendered_images, target_images)
    
    # Depth metrics
    if rendered_depths is not None and target_depths is not None:
        depth_metrics = compute_depth_accuracy(rendered_depths, target_depths)
        metrics.update(depth_metrics)
    
    # Pose metrics
    if estimated_poses is not None and ground_truth_poses is not None:
        pose_metrics = compute_pose_accuracy(estimated_poses, ground_truth_poses)
        metrics.update(pose_metrics)
    
    # Gaussian quality metrics
    if gaussians is not None:
        gaussian_metrics = compute_gaussian_quality_metrics(gaussians)
        metrics.update(gaussian_metrics)
    
    return metrics


def evaluate_on_dataset(model, dataset, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Move to device
            if isinstance(sample, dict):
                for key in sample:
                    if isinstance(sample[key], torch.Tensor):
                        sample[key] = sample[key].unsqueeze(0).to(device)
            
            # Forward pass
            if hasattr(model, 'forward'):
                output = model(sample)
            else:
                # Assume it's a tuple of (gaussian_model, pose_estimator, renderer)
                gaussian_model, pose_estimator, renderer = model
                gaussians = gaussian_model()
                poses = pose_estimator(sample['camera_id'])
                output = renderer(gaussians, poses)
            
            # Compute metrics
            metrics = compute_reconstruction_metrics(
                output, sample['image'],
                gaussians=gaussians if 'gaussians' in locals() else None
            )
            all_metrics.append(metrics)
    
    # Average metrics across dataset
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics 