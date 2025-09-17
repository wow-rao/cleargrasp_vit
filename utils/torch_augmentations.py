#
# cleargrasp_vit/utils/torch_augmentations.py
#
import torch
import torch.nn as nn
import random
from typing import List, Any, Tuple, Dict

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

class Resize:
    """Resizes all image-like tensors in a sample dictionary to a specified size."""
    def __init__(self, size: Tuple[int, int]):
        self.output_size = size
      
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Use the RGB image's shape to determine the original size
        h, w = sample['rgb'].shape[-2:]

        if isinstance(self.output_size, int):
            if h < w:
                new_h = self.output_size
                new_w = int(new_h * w / h)
            else:
                new_w = self.output_size
                new_h = int(new_w * h / w)
            size = (new_h, new_w)
        else:
            size = self.output_size

        for key, value in sample.items():
            # Apply to any item that is a tensor and has spatial dimensions
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                # Ensure tensor is in (C, H, W) format before unsqueezing
                if value.dim() == 2:
                    value = value.unsqueeze(0)  # Add channel dim: (H, W) -> (1, H, W)

                # --- FIX for ValueError ---
                # F.interpolate requires a 4D batch input (N, C, H, W).
                # We add a batch dimension, resize, and then remove it.
                batched_value = value.unsqueeze(0)

                # Use 'nearest' interpolation for masks and integer-based data
                # to avoid creating interpolated float values. Use 'bilinear' for others.
                mode = 'nearest' if 'mask' in key or 'boundary' in key else 'bilinear'
                
                try:
                  resized_tensor = nn.functional.interpolate(
                    batched_value,
                    size=size,
                    mode=mode,
                    align_corners=False if mode == 'bilinear' else None
                  )
                except:
                  print(batched_value.shape)
                  return None

                sample[key] = resized_tensor.squeeze(0) # Remove batch dimension

        return sample

class Normalize:
    """Normalizes the RGB image tensor with mean and standard deviation."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        rgb_tensor = sample['rgb'].to(self.mean.device)
        rgb_tensor = (rgb_tensor.float() / 255.0 - self.mean) / self.std
        sample['rgb'] = rgb_tensor
        return sample

class RandomHorizontalFlip:
    """Horizontally flip all images in the sample randomly with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    flipped_tensor = torch.flip(value, dims=[-1])
                    # Flip the x-component of the normal vector
                    if 'normals' in key:
                        flipped_tensor[0, :, :] *= -1
                    sample[key] = flipped_tensor
        return sample
