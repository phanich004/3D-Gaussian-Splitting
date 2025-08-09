"""
Data loading utilities for 3D Gaussian Splatting
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from typing import Optional, List, Dict, Tuple
import json


class ImageDataset(Dataset):
    """
    Dataset for loading images for 3D Gaussian Splatting training.
    Supports loading images from a directory and optionally camera poses.
    """
    
    def __init__(self, image_dir: str, image_size: int = 1024, 
                 pose_file: Optional[str] = None, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset
        
        Args:
            image_dir: Directory containing images
            image_size: Target image size (assumes square images)
            pose_file: Optional path to camera poses file
            transform: Optional custom transforms
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.pose_file = pose_file
        
        # Get image files
        self.image_files = self._get_image_files()
        
        # Load camera poses if available
        self.camera_poses = self._load_camera_poses()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files in the directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in sorted(os.listdir(self.image_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.image_dir, filename))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")
        
        print(f"Found {len(image_files)} images in {self.image_dir}")
        return image_files
    
    def _load_camera_poses(self) -> Optional[Dict]:
        """Load camera poses from file if available"""
        if self.pose_file is None or not os.path.exists(self.pose_file):
            print("No camera poses file provided or found")
            return None
        
        try:
            with open(self.pose_file, 'r') as f:
                poses = json.load(f)
            print(f"Loaded camera poses from {self.pose_file}")
            return poses
        except Exception as e:
            print(f"Failed to load camera poses: {e}")
            return None
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms for image preprocessing"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: (3, H, W) Image tensor
                - camera_id: (1,) Camera ID
                - pose: (4, 4) Camera pose matrix (if available)
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Create sample
        sample = {
            'image': image,
            'camera_id': torch.tensor([idx], dtype=torch.long),
            'image_path': image_path
        }
        
        # Add camera pose if available
        if self.camera_poses is not None:
            pose = self.camera_poses.get(f"camera_{idx}", self.camera_poses.get(str(idx)))
            if pose is not None:
                sample['pose'] = torch.tensor(pose, dtype=torch.float32)
        
        return sample


class MultiViewDataset(Dataset):
    """
    Dataset for multi-view images with camera poses.
    This is useful for training with known camera poses.
    """
    
    def __init__(self, image_dir: str, pose_file: str, image_size: int = 1024):
        """
        Initialize multi-view dataset
        
        Args:
            image_dir: Directory containing images
            pose_file: Path to camera poses file
            image_size: Target image size
        """
        self.image_dir = image_dir
        self.pose_file = pose_file
        self.image_size = image_size
        
        # Load data
        self.images, self.poses = self._load_data()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _load_data(self) -> Tuple[List[str], List[torch.Tensor]]:
        """Load images and poses"""
        with open(self.pose_file, 'r') as f:
            pose_data = json.load(f)
        
        images = []
        poses = []
        
        for item in pose_data:
            if 'image_path' in item and 'pose' in item:
                image_path = os.path.join(self.image_dir, item['image_path'])
                if os.path.exists(image_path):
                    images.append(image_path)
                    poses.append(torch.tensor(item['pose'], dtype=torch.float32))
        
        return images, poses
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'pose': self.poses[idx],
            'camera_id': torch.tensor([idx], dtype=torch.long)
        }


class SparseViewDataset(Dataset):
    """
    Dataset specifically designed for sparse view reconstruction.
    This dataset supports loading only a few views per scene.
    """
    
    def __init__(self, image_dir: str, num_views: int = 5, image_size: int = 1024):
        """
        Initialize sparse view dataset
        
        Args:
            image_dir: Directory containing images
            num_views: Number of views to use (randomly sampled)
            image_size: Target image size
        """
        self.image_dir = image_dir
        self.num_views = num_views
        self.image_size = image_size
        
        # Get all image files
        self.all_image_files = self._get_image_files()
        
        # Randomly sample views
        if len(self.all_image_files) > num_views:
            indices = np.random.choice(len(self.all_image_files), num_views, replace=False)
            self.image_files = [self.all_image_files[i] for i in sorted(indices)]
        else:
            self.image_files = self.all_image_files
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Using {len(self.image_files)} sparse views out of {len(self.all_image_files)} total images")
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in sorted(os.listdir(self.image_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.image_dir, filename))
        
        return image_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'camera_id': torch.tensor([idx], dtype=torch.long),
            'image_path': image_path
        }


def create_data_loader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True, 
                      num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the given dataset
    
    Args:
        dataset: Dataset to wrap
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples
    """
    # Extract components
    images = torch.stack([item['image'] for item in batch])
    camera_ids = torch.cat([item['camera_id'] for item in batch])
    
    # Handle optional pose data
    if 'pose' in batch[0]:
        poses = torch.stack([item['pose'] for item in batch])
    else:
        poses = None
    
    result = {
        'images': images,
        'camera_ids': camera_ids
    }
    
    if poses is not None:
        result['poses'] = poses
    
    return result 