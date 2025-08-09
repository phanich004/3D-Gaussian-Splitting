#!/usr/bin/env python3
"""
Main training script for No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import wandb
from tqdm import tqdm
import numpy as np

from models.gaussian_model import GaussianModel
from models.pose_estimator import PoseEstimator
from models.renderer import GaussianRenderer
from losses.photometric_loss import PhotometricLoss
from losses.geometric_loss import GeometricLoss
from losses.regularization import RegularizationLoss
from utils.data_loader import ImageDataset
from utils.visualization import visualize_gaussians, save_rendered_images
from utils.metrics import compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Pose-Free 3D Gaussian Splatting")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--num_iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_gaussians", type=int, default=100000, help="Number of 3D Gaussians")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(args):
    """Setup logging with wandb if enabled"""
    if args.wandb:
        wandb.init(project="pose-free-3d-gaussian", config=vars(args))
    return wandb if args.wandb else None


def create_models(config, device):
    """Create and initialize all models"""
    # Initialize Gaussian model
    gaussian_model = GaussianModel(
        num_gaussians=config['num_gaussians'],
        device=device
    ).to(device)
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        num_cameras=config.get('num_cameras', 100),
        device=device
    ).to(device)
    
    # Initialize renderer
    renderer = GaussianRenderer(
        image_size=config['image_size'],
        device=device
    ).to(device)
    
    return gaussian_model, pose_estimator, renderer


def create_losses(config):
    """Create loss functions"""
    photometric_loss = PhotometricLoss(
        loss_type=config['photometric_loss']['type'],
        weight=config['photometric_loss']['weight']
    )
    
    geometric_loss = GeometricLoss(
        loss_type=config['geometric_loss']['type'],
        weight=config['geometric_loss']['weight']
    )
    
    regularization_loss = RegularizationLoss(
        sparsity_weight=config['regularization']['sparsity_weight'],
        smoothness_weight=config['regularization']['smoothness_weight']
    )
    
    return photometric_loss, geometric_loss, regularization_loss


def training_step(gaussian_model, pose_estimator, renderer, 
                 photometric_loss, geometric_loss, regularization_loss,
                 images, camera_ids, optimizer_gaussian, optimizer_pose):
    """Single training step"""
    
    # Forward pass
    gaussians = gaussian_model()
    poses = pose_estimator(camera_ids)
    
    # Render images
    rendered_images = renderer(gaussians, poses)
    
    # Compute losses
    photo_loss = photometric_loss(rendered_images, images)
    geo_loss = geometric_loss(gaussians, poses)
    reg_loss = regularization_loss(gaussians)
    
    total_loss = photo_loss + geo_loss + reg_loss
    
    # Backward pass
    optimizer_gaussian.zero_grad()
    optimizer_pose.zero_grad()
    total_loss.backward()
    
    optimizer_gaussian.step()
    optimizer_pose.step()
    
    return {
        'total_loss': total_loss.item(),
        'photometric_loss': photo_loss.item(),
        'geometric_loss': geo_loss.item(),
        'regularization_loss': reg_loss.item()
    }


def save_checkpoint(gaussian_model, pose_estimator, optimizer_gaussian, 
                   optimizer_pose, iteration, output_dir):
    """Save training checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'gaussian_model_state_dict': gaussian_model.state_dict(),
        'pose_estimator_state_dict': pose_estimator.state_dict(),
        'optimizer_gaussian_state_dict': optimizer_gaussian.state_dict(),
        'optimizer_pose_state_dict': optimizer_pose.state_dict(),
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{iteration:06d}.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = ImageDataset(args.input_dir, image_size=config['image_size'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create models
    gaussian_model, pose_estimator, renderer = create_models(config, device)
    
    # Create losses
    photometric_loss, geometric_loss, regularization_loss = create_losses(config)
    
    # Create optimizers
    optimizer_gaussian = optim.Adam(gaussian_model.parameters(), lr=args.learning_rate)
    optimizer_pose = optim.Adam(pose_estimator.parameters(), lr=args.learning_rate)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
        pose_estimator.load_state_dict(checkpoint['pose_estimator_state_dict'])
        optimizer_gaussian.load_state_dict(checkpoint['optimizer_gaussian_state_dict'])
        optimizer_pose.load_state_dict(checkpoint['optimizer_pose_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    print("Starting training...")
    for iteration in tqdm(range(start_iteration, args.num_iterations)):
        
        # Get batch data
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4))
            batch = next(iter(dataloader))
        
        images = batch['images'].to(device)
        camera_ids = batch['camera_ids'].to(device)
        
        # Training step
        losses = training_step(
            gaussian_model, pose_estimator, renderer,
            photometric_loss, geometric_loss, regularization_loss,
            images, camera_ids, optimizer_gaussian, optimizer_pose
        )
        
        # Logging
        if logger and iteration % 100 == 0:
            logger.log(losses, step=iteration)
        
        # Print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Total Loss = {losses['total_loss']:.4f}")
            
            # Save checkpoint
            if iteration % 5000 == 0:
                save_checkpoint(gaussian_model, pose_estimator, optimizer_gaussian, 
                              optimizer_pose, iteration, args.output_dir)
                
                # Visualize results
                with torch.no_grad():
                    gaussians = gaussian_model()
                    poses = pose_estimator(torch.arange(len(dataset)).to(device))
                    rendered_images = renderer(gaussians, poses)
                    
                    # Save visualization
                    save_rendered_images(rendered_images, os.path.join(args.output_dir, f'render_{iteration:06d}.png'))
    
    # Save final model
    save_checkpoint(gaussian_model, pose_estimator, optimizer_gaussian, 
                   optimizer_pose, args.num_iterations - 1, args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main() 