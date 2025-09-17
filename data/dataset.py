#
# cleargrasp_vit/data/dataset.py
#
import os
import glob
from torch.utils.data import Dataset
from..utils.torch_image_io import read_png_torch
from..utils.torch_augmentations import Compose

class ClearGraspViT_Dataset(Dataset):
    """
    Custom PyTorch Dataset for the ClearGrasp dataset, using pure torch I/O.
    """
    def __init__(self, root_dir: str, subset: str = 'all', split_ratio: float = 0.7,
                 transform: Compose = None, device: str = 'cpu'):
        """
        Args:
            root_dir (str): Path to the ClearGrasp dataset directory.
            subset (str): Which subset to use ('encoder', 'decoder', or 'all').
            split_ratio (float): The ratio to split data for encoder training.
            transform (Compose): Augmentation pipeline.
            device (str): Device to load tensors onto.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        
        # Find all synthetic data samples
        rgb_files = sorted(glob.glob(os.path.join(root_dir, '*-rgb.png')))
        
        # Create a list of sample identifiers
        self.samples = [f.replace('-rgb.png', '') for f in rgb_files]
        
        # Split dataset for two-stage training
        split_index = int(len(self.samples) * split_ratio)
        if subset == 'encoder':
            self.samples = self.samples[:split_index]
        elif subset == 'decoder':
            self.samples = self.samples[split_index:]
        elif subset!= 'all':
            raise ValueError("subset must be one of 'encoder', 'decoder', or 'all'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_prefix = self.samples[idx]
        
        # Load all data modalities into torch tensors
        rgb_image = read_png_torch(f"{sample_prefix}-rgb.png", self.device)
        depth_image = read_png_torch(f"{sample_prefix}-depth.png", self.device)
        normals_gt = read_png_torch(f"{sample_prefix}-normal.png", self.device)
        mask_gt = read_png_torch(f"{sample_prefix}-mask.png", self.device)
        boundary_gt = read_png_torch(f"{sample_prefix}-boundary.png", self.device)
        
        # The ground truth depth is in a different folder for real-world data
        gt_depth_path = f"{sample_prefix.replace('transparent', 'gt')}-depth.png"
        if os.path.exists(gt_depth_path):
            depth_gt = read_png_torch(gt_depth_path, self.device)
        else: # For synthetic data, it might be named differently or not exist
            depth_gt = torch.zeros_like(depth_image) # Placeholder

        # Apply transformations if any
        if self.transform:
            # Note: The transform pipeline needs to be designed to handle all these inputs
            rgb_image, normals_gt, depth_image, mask_gt, boundary_gt, depth_gt = self.transform(
                rgb_image, normals_gt, depth_image, mask_gt, boundary_gt, depth_gt
            )
            
        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'normals_gt': normals_gt,
            'mask_gt': mask_gt,
            'boundary_gt': boundary_gt,
            'depth_gt': depth_gt
        }
