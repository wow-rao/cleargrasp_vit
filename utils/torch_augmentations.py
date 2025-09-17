#
# cleargrasp_vit/utils/torch_augmentations.py
#
import torch
import random
from typing import List, Any, Tuple

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

class ToTensor:
    """Converts numpy array to torch tensor. Assumes input is HWC."""
    def __call__(self, image, *other_data):
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return (tensor,) + other_data

class Normalize:
    """Normalizes a tensor image with mean and standard deviation."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor, *other_data):
        tensor = (tensor.float() / 255.0 - self.mean) / self.std
        return (tensor,) + other_data

class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, normals: torch.Tensor, *other_data):
        if random.random() < self.p:
            image = torch.flip(image, )
            normals = torch.flip(normals, )
            normals[0, :, :] *= -1 # Flip the x-component of the normal vector
        return (image, normals) + other_data

class RandomRotation:
    """Rotate the image by a random angle."""
    def __init__(self, degrees: Tuple[float, float]):
        self.degrees = degrees

    def __call__(self, image: torch.Tensor, *other_data):
        angle = random.uniform(self.degrees, self.degrees)
        theta = torch.tensor([torch.cos(torch.tensor(angle * torch.pi / 180.0)), -torch.sin(torch.tensor(angle * torch.pi / 180.0)), 0],
            [torch.sin(torch.tensor(angle * torch.pi / 180.0)), torch.cos(torch.tensor(angle * torch.pi / 180.0)), 0], dtype=torch.float32)
        
        grid = nn.functional.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False).to(image.device)
        
        rotated_image = nn.functional.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
        
        # Apply rotation to other tensors if they are provided
        rotated_others =
        for data in other_data:
            if isinstance(data, torch.Tensor) and data.dim() >= 3:
                rotated_data = nn.functional.grid_sample(data.unsqueeze(0), grid, align_corners=False).squeeze(0)
                rotated_others.append(rotated_data)
            else:
                rotated_others.append(data)

        return (rotated_image,) + tuple(rotated_others)

class ColorJitter:
    """Randomly change the brightness, contrast, saturation and hue of an image."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: torch.Tensor, *other_data):
        # This is a simplified implementation. A full implementation would require
        # converting to HSV for hue/saturation, which is complex in pure torch.
        # Here we demonstrate brightness and contrast.
        
        # Brightness
        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = image.float() * factor
            image = torch.clamp(image, 0, 255).to(torch.uint8)

        # Contrast
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = torch.mean(image.float(), dim=(1, 2), keepdim=True)
            image = (image.float() - mean) * factor + mean
            image = torch.clamp(image, 0, 255).to(torch.uint8)
            
        return (image,) + other_data
