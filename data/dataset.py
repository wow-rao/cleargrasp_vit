#
# cleargrasp_vit/data/dataset.py
#

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_folder_path = os.path.join(parent_dir, 'utils')

sys.path.insert(0, target_folder_path)

import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch_image_io import read_png_torch, read_jpg_torch, read_exr_torch
from typing import Dict, Any, List

class ClearGraspViT_Dataset(Dataset):
    """
    Custom PyTorch Dataset for the ClearGrasp training dataset.
    This class handles the complex, multi-level directory structure and varied
    file formats (.jpg,.png,.exr) of the official training data.
    """
    def __init__(self, root_dir: str, subset: str = 'all', split_ratio: float = 0.7,
                 transform: transforms.Compose = None, device: str = 'cpu'):
        """
        Args:
            root_dir (str): Path to the parent directory, e.g., '.../cleargrasp-dataset-train'.
            subset (str): Which subset to use ('encoder', 'decoder', or 'all').
            split_ratio (float): The ratio to split data for encoder training.
            transform (Compose): Augmentation and preprocessing pipeline.
            device (str): Device to load tensors onto ('cpu' or 'cuda').
        """
        self.root_dir = './data/cleargrasp-dataset-train'
        self.transform = transform
        self.device = device
        self.samples = self._find_samples()

        if not self.samples:
            raise FileNotFoundError(f"No valid samples found in {root_dir}. Please check the directory structure.")

        # Split dataset for two-stage training
        split_index = int(len(self.samples) * split_ratio)
        if subset == 'encoder':
            self.samples = self.samples[:split_index]
        elif subset == 'decoder':
            self.samples = self.samples[split_index:]
        elif subset!= 'all':
            raise ValueError("subset must be one of 'encoder', 'decoder', or 'all'")

    def _find_samples(self):
        """Scans the directory structure to find all corresponding data files for each sample."""
        sample_list = []
        # Find all object subdirectories (e.g., 'cup-with-waves-train')
        object_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

        for obj_dir in object_dirs:
            rgb_dir = os.path.join(self.root_dir, obj_dir, 'rgb-imgs')
            if not os.path.isdir(rgb_dir):
                continue

            # Use rgb images as the source of truth for sample names
            for rgb_filename in sorted(os.listdir(rgb_dir)):
                base_name, ext = os.path.splitext(rgb_filename)
                base_name = base_name.split('-')[0]
                if ext.lower()!= '.jpg':
                    continue

                # Construct paths for all modalities
                paths = {
                    'rgb': os.path.join(rgb_dir, f"{base_name}-rgb.jpg"),
                    'depth': os.path.join(self.root_dir, obj_dir, 'depth-imgs-rectified', f"{base_name}-depth-rectified.exr"),
                    'normals_gt': os.path.join(self.root_dir, obj_dir, 'camera-normals', f"{base_name}-cameraNormals.exr"),
                    'mask_gt': os.path.join(self.root_dir, obj_dir, 'segmentation-masks', f"{base_name}-segmentation-mask.png"),
                    'boundary_gt': os.path.join(self.root_dir, obj_dir, 'outlines', f"{base_name}-outlineSegmentation.png"),
                }

                # Ensure all required files for this sample exist
                if all(os.path.exists(p) for p in paths.values()):
                    sample_list.append(paths)
        
        return sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_paths = self.samples[idx]
        
        # --- Load all data modalities using the appropriate reader ---
        rgb = read_jpg_torch(sample_paths['rgb'], self.device)
        depth_raw = read_exr_torch(sample_paths['depth'], self.device)
        normals_gt = read_exr_torch(sample_paths['normals_gt'], self.device, True)
        mask_gt = read_png_torch(sample_paths['mask_gt'], self.device)
        mask_gt = mask_gt / 255
        
        # Boundary is a JPG but should be treated as a single-channel mask
        boundary_jpg = read_png_torch(sample_paths['boundary_gt'], self.device)
        boundary_gt = (boundary_jpg > 128).float().unsqueeze(0) # Take one channel, threshold, and add channel dim back

        # For synthetic training data, the rendered depth is the ground truth
        depth_gt = depth_raw.clone()

        sample = {
            'rgb': rgb,
            'depth': depth_raw,
            'normals_gt': normals_gt,
            'mask_gt': mask_gt,
            'boundary_gt': boundary_gt,
            'depth_gt': depth_gt,
            'name': os.path.splitext(os.path.basename(sample_paths['rgb']))
        }

        if self.transform:
            for key, value in sample.items():
              if key == 'name':
                continue
              sample[key] = self.transform(value)
            
        return sample
