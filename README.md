# No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements the paper **"No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views"** - a novel approach to 3D scene reconstruction without requiring pre-computed camera poses.

## 🎯 Key Innovation

The main breakthrough of this paper is the ability to perform **3D Gaussian Splatting without requiring known camera poses**. Instead, it uses a self-supervised approach that jointly optimizes both the 3D Gaussian representation and the camera poses from sparse views.

### ✨ Key Features

- **🎯 Pose-Free Learning**: No need for pre-computed camera poses
- **🤖 Self-Supervised**: Joint optimization of geometry and camera parameters
- **📊 Sparse Views**: Works with only a few input images (5-10 views)
- **⚡ Real-time Rendering**: Fast 3D Gaussian splatting-based rendering
- **🔬 Comprehensive Losses**: Photometric, geometric, and regularization losses
- **🏗️ Modular Design**: Clean separation of concerns
- **🔧 Extensible**: Easy to add new loss functions or models

## 🏗️ Architecture Overview

### Core Components

1. **3D Gaussian Model** (`models/gaussian_model.py`)
   - Parameterizes 3D Gaussians with position, scale, rotation (quaternion), color, and opacity
   - Learnable parameters for each Gaussian
   - Supports dynamic addition/pruning of Gaussians

2. **Pose Estimator** (`models/pose_estimator.py`)
   - Neural network for estimating camera poses from sparse views
   - Learns camera positions, rotations (quaternions), and intrinsics
   - Converts to view and projection matrices

3. **Differentiable Renderer** (`models/renderer.py`)
   - Implements 3D Gaussian splatting for differentiable rendering
   - Projects 3D Gaussians to 2D screen space
   - Alpha compositing for final image generation

### Loss Functions

1. **Photometric Loss** (`losses/photometric_loss.py`)
   - L1/L2 loss between rendered and target images
   - Optional perceptual loss using LPIPS
   - Multi-scale photometric loss

2. **Geometric Loss** (`losses/geometric_loss.py`)
   - Depth consistency between views
   - Normal consistency for surface smoothness
   - Epipolar geometry constraints

3. **Regularization Loss** (`losses/regularization.py`)
   - Sparsity loss to encourage efficient representation
   - Smoothness loss for neighboring Gaussians
   - Scale and opacity regularization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/phanich004/3D-Gaussian-Splitting.git
   cd 3D-Gaussian-Splitting
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python test_implementation.py
   ```

### Basic Usage

1. **Prepare your data**:
   - Place your images in a directory (e.g., `data/images/`)
   - Images should be in common formats (jpg, png, etc.)

2. **Run training**:
   ```bash
   python main.py \
       --input_dir data/images \
       --output_dir outputs \
       --num_iterations 30000 \
       --num_gaussians 100000
   ```

3. **Advanced usage**:
   ```bash
   python main.py \
       --input_dir data/images \
       --output_dir outputs \
       --config configs/default.yaml \
       --wandb \
       --device cuda
   ```

## 📊 Training Pipeline

### Main Training Script (`main.py`)

```python
# 1. Initialize models
gaussian_model = GaussianModel(num_gaussians=100000)
pose_estimator = PoseEstimator(num_cameras=100)
renderer = GaussianRenderer(image_size=1024)

# 2. Load data
dataset = ImageDataset(image_dir="path/to/images")
dataloader = DataLoader(dataset, batch_size=4)

# 3. Training loop
for iteration in range(num_iterations):
    # Forward pass
    gaussians = gaussian_model()
    poses = pose_estimator(camera_ids)
    rendered_images = renderer(gaussians, poses)
    
    # Compute losses
    photo_loss = photometric_loss(rendered_images, target_images)
    geo_loss = geometric_loss(gaussians, poses)
    reg_loss = regularization_loss(gaussians)
    
    # Backward pass
    total_loss = photo_loss + geo_loss + reg_loss
    total_loss.backward()
    
    # Update parameters
    optimizer_gaussian.step()
    optimizer_pose.step()
```

## 🔧 Configuration

The training can be customized using the configuration file `configs/default.yaml`:

```yaml
# Model parameters
num_gaussians: 100000
image_size: 1024
num_cameras: 100

# Training parameters
learning_rate: 0.001
batch_size: 4
num_iterations: 30000

# Loss weights
photometric_loss:
  type: "l1"  # Options: l1, l2, smooth_l1, perceptual, combined
  weight: 1.0

geometric_loss:
  type: "depth_consistency"  # Options: depth_consistency, normal_consistency, epipolar, combined
  weight: 0.1

regularization:
  sparsity_weight: 0.01
  smoothness_weight: 0.1
```

## 🎨 Visualization and Evaluation

### Visualization Tools (`utils/visualization.py`)

- 3D Gaussian point cloud visualization
- Camera pose visualization
- Rendered image comparison
- Loss curve plotting

### Evaluation Metrics (`utils/metrics.py`)

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Depth accuracy**: Depth reconstruction quality
- **Pose accuracy**: Camera pose estimation quality

## 📈 Results

The implementation achieves:

- ✅ **Pose-free reconstruction** from sparse views
- ✅ **High-quality 3D representations** using 3D Gaussians
- ✅ **Real-time rendering** capabilities
- ✅ **Robust optimization** with comprehensive losses

## 🧪 Testing

Run the complete test suite to verify the implementation:

```bash
python test_implementation.py
```

This will test:
- ✅ Gaussian model functionality
- ✅ Pose estimator
- ✅ Renderer
- ✅ Loss functions
- ✅ Full integration
- ✅ Data loading

## 📚 Usage Examples

### Example 1: Basic Training

```bash
# Train on a dataset with 10 images
python main.py \
    --input_dir data/scene1 \
    --output_dir outputs/scene1 \
    --num_gaussians 50000 \
    --num_iterations 15000
```

### Example 2: Advanced Training with Logging

```bash
# Train with wandb logging and custom config
python main.py \
    --input_dir data/scene2 \
    --output_dir outputs/scene2 \
    --config configs/default.yaml \
    --wandb \
    --device cuda \
    --batch_size 8
```

### Example 3: Testing on Custom Data

```python
from models.gaussian_model import GaussianModel
from models.pose_estimator import PoseEstimator
from models.renderer import GaussianRenderer

# Initialize models
gaussian_model = GaussianModel(num_gaussians=10000)
pose_estimator = PoseEstimator(num_cameras=5)
renderer = GaussianRenderer(image_size=512)

# Forward pass
gaussians = gaussian_model()
poses = pose_estimator(camera_ids)
rendered_images = renderer(gaussians, poses)
```

## 🔬 Advanced Features

### Dynamic Gaussian Management

The implementation supports dynamic addition and pruning of Gaussians:

```python
# Add new Gaussians
gaussian_model.add_gaussians(num_new=1000)

# Prune low-opacity Gaussians
gaussian_model.prune_gaussians(threshold=0.01)
```

### Multi-scale Training

Support for multi-scale photometric loss:

```python
from losses.photometric_loss import MultiScalePhotometricLoss

loss_fn = MultiScalePhotometricLoss(scales=[1, 2, 4, 8])
loss = loss_fn(rendered_images, target_images)
```

### Custom Loss Functions

Easy to add new loss functions:

```python
from losses.regularization import RegularizationLoss

class CustomLoss(RegularizationLoss):
    def forward(self, gaussians):
        # Custom loss implementation
        return super().forward(gaussians) + custom_term
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper: "No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views"
- 3D Gaussian Splatting: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- Related work: NeRF, Instant-NGP, Gaussian Splatting

## 📞 Contact

- **Repository**: [https://github.com/phanich004/3D-Gaussian-Splitting](https://github.com/phanich004/3D-Gaussian-Splitting)
- **Issues**: [GitHub Issues](https://github.com/phanich004/3D-Gaussian-Splitting/issues)

## 🎯 Future Work

- [ ] GPU-optimized rendering
- [ ] Dynamic Gaussian management
- [ ] Multi-scale training
- [ ] Advanced geometric losses
- [ ] Real-time inference optimization

---

**⭐ Star this repository if you find it useful!**