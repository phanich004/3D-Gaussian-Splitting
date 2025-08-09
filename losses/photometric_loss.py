"""
Photometric loss functions for 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhotometricLoss(nn.Module):
    """
    Photometric loss for measuring reconstruction quality between rendered and target images.
    This includes L1 loss, L2 loss, and perceptual loss components.
    """
    
    def __init__(self, loss_type: str = 'l1', weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        
        # Initialize perceptual loss if needed
        if 'perceptual' in loss_type:
            self.perceptual_loss = self._init_perceptual_loss()
    
    def _init_perceptual_loss(self):
        """Initialize VGG-based perceptual loss"""
        try:
            import lpips
            return lpips.LPIPS(net='vgg')
        except ImportError:
            print("Warning: LPIPS not available. Using L1 loss instead.")
            return None
    
    def forward(self, rendered_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute photometric loss between rendered and target images
        
        Args:
            rendered_images: (B, H, W, 3) Rendered images
            target_images: (B, H, W, 3) Target images
            
        Returns:
            torch.Tensor: Photometric loss
        """
        # Ensure images are in the correct format
        if rendered_images.shape[-1] == 3:
            rendered_images = rendered_images.permute(0, 3, 1, 2)  # (B, 3, H, W)
        if target_images.shape[-1] == 3:
            target_images = target_images.permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        # Normalize to [0, 1] if needed
        if rendered_images.max() > 1.0:
            rendered_images = rendered_images / 255.0
        if target_images.max() > 1.0:
            target_images = target_images / 255.0
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(rendered_images, target_images)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(rendered_images, target_images)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(rendered_images, target_images)
        elif self.loss_type == 'perceptual':
            loss = self._compute_perceptual_loss(rendered_images, target_images)
        elif self.loss_type == 'combined':
            l1_loss = F.l1_loss(rendered_images, target_images)
            perceptual_loss = self._compute_perceptual_loss(rendered_images, target_images)
            loss = 0.5 * l1_loss + 0.5 * perceptual_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.weight * loss
    
    def _compute_perceptual_loss(self, rendered_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features
        
        Args:
            rendered_images: (B, 3, H, W) Rendered images
            target_images: (B, 3, H, W) Target images
            
        Returns:
            torch.Tensor: Perceptual loss
        """
        if self.perceptual_loss is None:
            # Fallback to L1 loss
            return F.l1_loss(rendered_images, target_images)
        
        # Normalize images to [-1, 1] for LPIPS
        rendered_normalized = 2 * rendered_images - 1
        target_normalized = 2 * target_images - 1
        
        # Compute perceptual loss
        perceptual_loss = self.perceptual_loss(rendered_normalized, target_normalized)
        return torch.mean(perceptual_loss)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for measuring structural similarity between images.
    """
    
    def __init__(self, window_size: int = 11, weight: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.weight = weight
    
    def forward(self, rendered_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss
        
        Args:
            rendered_images: (B, 3, H, W) Rendered images
            target_images: (B, 3, H, W) Target images
            
        Returns:
            torch.Tensor: SSIM loss (1 - SSIM)
        """
        # Convert SSIM to loss (1 - SSIM)
        ssim = self._compute_ssim(rendered_images, target_images)
        return self.weight * (1 - ssim)
    
    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images
        
        Args:
            x: (B, 3, H, W) First image
            y: (B, 3, H, W) Second image
            
        Returns:
            torch.Tensor: SSIM values
        """
        # Compute means
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size//2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size//2)
        
        # Compute variances and covariance
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size//2) - mu_xy
        
        # SSIM parameters
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM formula
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return torch.mean(ssim)


class MultiScalePhotometricLoss(nn.Module):
    """
    Multi-scale photometric loss that computes loss at multiple scales.
    This helps with both fine details and global structure.
    """
    
    def __init__(self, scales: list = [1, 2, 4, 8], loss_type: str = 'l1', weight: float = 1.0):
        super().__init__()
        self.scales = scales
        self.loss_type = loss_type
        self.weight = weight
        
        self.base_loss = PhotometricLoss(loss_type=loss_type)
    
    def forward(self, rendered_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale photometric loss
        
        Args:
            rendered_images: (B, 3, H, W) Rendered images
            target_images: (B, 3, H, W) Target images
            
        Returns:
            torch.Tensor: Multi-scale photometric loss
        """
        total_loss = 0.0
        
        for scale in self.scales:
            if scale == 1:
                # Original scale
                scaled_rendered = rendered_images
                scaled_target = target_images
            else:
                # Downsample images
                scaled_rendered = F.interpolate(rendered_images, 
                                             scale_factor=1.0/scale, 
                                             mode='bilinear', 
                                             align_corners=False)
                scaled_target = F.interpolate(target_images, 
                                            scale_factor=1.0/scale, 
                                            mode='bilinear', 
                                            align_corners=False)
            
            # Compute loss at this scale
            scale_loss = self.base_loss(scaled_rendered, scaled_target)
            total_loss += scale_loss / len(self.scales)
        
        return self.weight * total_loss 