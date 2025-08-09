#!/usr/bin/env python3
"""
Test script for Pose-Free 3D Gaussian Splatting implementation
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gaussian_model import GaussianModel
from models.pose_estimator import PoseEstimator
from models.renderer import GaussianRenderer
from losses.photometric_loss import PhotometricLoss
from losses.geometric_loss import GeometricLoss
from losses.regularization import RegularizationLoss
from utils.data_loader import ImageDataset


def test_gaussian_model():
    """Test Gaussian model"""
    print("Testing Gaussian model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussian_model = GaussianModel(num_gaussians=1000, device=device)
    
    # Test forward pass
    gaussians = gaussian_model()
    
    # Check shapes
    assert gaussians['positions'].shape == (1000, 3), f"Expected (1000, 3), got {gaussians['positions'].shape}"
    assert gaussians['scales'].shape == (1000, 3), f"Expected (1000, 3), got {gaussians['scales'].shape}"
    assert gaussians['rotations'].shape == (1000, 4), f"Expected (1000, 4), got {gaussians['rotations'].shape}"
    assert gaussians['colors'].shape == (1000, 3), f"Expected (1000, 3), got {gaussians['colors'].shape}"
    assert gaussians['opacities'].shape == (1000, 1), f"Expected (1000, 1), got {gaussians['opacities'].shape}"
    
    # Check value ranges
    assert torch.all(gaussians['scales'] > 0), "Scales should be positive"
    assert torch.all(gaussians['colors'] >= 0) and torch.all(gaussians['colors'] <= 1), "Colors should be in [0, 1]"
    assert torch.all(gaussians['opacities'] >= 0) and torch.all(gaussians['opacities'] <= 1), "Opacities should be in [0, 1]"
    
    print("✓ Gaussian model test passed!")


def test_pose_estimator():
    """Test pose estimator"""
    print("Testing pose estimator...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_estimator = PoseEstimator(num_cameras=10, device=device)
    
    # Test forward pass
    camera_ids = torch.arange(5, device=device)
    poses = pose_estimator(camera_ids)
    
    # Check shapes
    assert poses['positions'].shape == (5, 3), f"Expected (5, 3), got {poses['positions'].shape}"
    assert poses['rotations'].shape == (5, 4), f"Expected (5, 4), got {poses['rotations'].shape}"
    assert poses['view_matrices'].shape == (5, 4, 4), f"Expected (5, 4, 4), got {poses['view_matrices'].shape}"
    assert poses['projection_matrices'].shape == (5, 4, 4), f"Expected (5, 4, 4), got {poses['projection_matrices'].shape}"
    
    print("✓ Pose estimator test passed!")


def test_renderer():
    """Test renderer"""
    print("Testing renderer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = GaussianRenderer(image_size=256, device=device)
    
    # Create dummy data
    gaussians = {
        'positions': torch.rand(100, 3, device=device) * 2 - 1,
        'scales': torch.ones(100, 3, device=device) * 0.1,
        'rotations': torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(100, 1),
        'colors': torch.rand(100, 3, device=device),
        'opacities': torch.ones(100, 1, device=device) * 0.5
    }
    
    poses = {
        'view_matrices': torch.eye(4, device=device).unsqueeze(0).repeat(1, 1, 1),
        'projection_matrices': torch.eye(4, device=device).unsqueeze(0).repeat(1, 1, 1)
    }
    
    # Test forward pass
    rendered_images = renderer(gaussians, poses)
    
    # Check shape
    assert rendered_images.shape == (1, 256, 256, 3), f"Expected (1, 256, 256, 3), got {rendered_images.shape}"
    
    print("✓ Renderer test passed!")


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test photometric loss
    photometric_loss = PhotometricLoss(loss_type='l1', weight=1.0)
    rendered_images = torch.rand(2, 3, 64, 64, device=device)
    target_images = torch.rand(2, 3, 64, 64, device=device)
    
    loss = photometric_loss(rendered_images, target_images)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test geometric loss
    geometric_loss = GeometricLoss(loss_type='depth_consistency', weight=1.0)
    gaussians = {
        'positions': torch.rand(100, 3, device=device),
        'opacities': torch.rand(100, 1, device=device)
    }
    poses = {
        'view_matrices': torch.eye(4, device=device).unsqueeze(0).repeat(2, 1, 1)
    }
    
    loss = geometric_loss(gaussians, poses)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    
    # Test regularization loss
    regularization_loss = RegularizationLoss(sparsity_weight=0.01, smoothness_weight=0.1)
    loss = regularization_loss(gaussians)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    
    print("✓ Loss functions test passed!")


def test_integration():
    """Test full integration"""
    print("Testing full integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    gaussian_model = GaussianModel(num_gaussians=1000, device=device)
    pose_estimator = PoseEstimator(num_cameras=5, device=device)
    renderer = GaussianRenderer(image_size=128, device=device)
    
    # Create losses
    photometric_loss = PhotometricLoss(loss_type='l1', weight=1.0)
    geometric_loss = GeometricLoss(loss_type='depth_consistency', weight=0.1)
    regularization_loss = RegularizationLoss(sparsity_weight=0.01, smoothness_weight=0.1)
    
    # Test forward pass
    gaussians = gaussian_model()
    camera_ids = torch.arange(2, device=device)
    poses = pose_estimator(camera_ids)
    rendered_images = renderer(gaussians, poses)
    
    # Test losses
    target_images = torch.rand(2, 128, 128, 3, device=device)
    photo_loss = photometric_loss(rendered_images, target_images)
    geo_loss = geometric_loss(gaussians, poses)
    reg_loss = regularization_loss(gaussians)
    
    total_loss = photo_loss + geo_loss + reg_loss
    
    # Test backward pass
    total_loss.backward()
    
    print("✓ Full integration test passed!")


def test_data_loader():
    """Test data loader"""
    print("Testing data loader...")
    
    # Create a dummy dataset directory if it doesn't exist
    dummy_dir = "test_data"
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Create dummy images
    from PIL import Image
    dummy_image = Image.new('RGB', (256, 256), color='red')
    dummy_image.save(os.path.join(dummy_dir, "test_image_1.png"))
    dummy_image.save(os.path.join(dummy_dir, "test_image_2.png"))
    
    try:
        dataset = ImageDataset(dummy_dir, image_size=128)
        assert len(dataset) == 2, f"Expected 2 images, got {len(dataset)}"
        
        # Test getting a sample
        sample = dataset[0]
        assert 'image' in sample, "Sample should contain 'image'"
        assert 'camera_id' in sample, "Sample should contain 'camera_id'"
        assert sample['image'].shape == (3, 128, 128), f"Expected (3, 128, 128), got {sample['image'].shape}"
        
        print("✓ Data loader test passed!")
        
    finally:
        # Clean up
        import shutil
        if os.path.exists(dummy_dir):
            shutil.rmtree(dummy_dir)


def main():
    """Run all tests"""
    print("Starting tests for Pose-Free 3D Gaussian Splatting implementation...")
    
    try:
        test_gaussian_model()
        test_pose_estimator()
        test_renderer()
        test_losses()
        test_integration()
        test_data_loader()
        
        print("\n🎉 All tests passed! Implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 