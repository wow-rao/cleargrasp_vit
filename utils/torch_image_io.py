#
# cleargrasp_vit/utils/torch_image_io.py
#
import torch
import png  # PyPNG library
import numpy as np
from typing import Union, Tuple

def read_png_torch(path: str, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Reads a PNG image file and returns it as a torch.Tensor.
    Handles RGB (8-bit) and Grayscale (16-bit for depth) images.

    Args:
        path (str): Path to the PNG file.
        device (Union[str, torch.device]): The device to place the tensor on.

    Returns:
        torch.Tensor: The image tensor. Shape (C, H, W) for RGB, (1, H, W) for grayscale.
    """
    reader = png.Reader(filename=path)
    width, height, rows, info = reader.read()
    
    # Convert iterator of rows to a flat list of pixel values
    image_data = np.vstack(list(map(np.uint8, rows)))
    
    if info['greyscale']:
        # Grayscale image (likely depth map)
        if info['bitdepth'] == 16:
            # Handle 16-bit depth data correctly
            image_data = image_data.astype(np.uint16)
        
        tensor = torch.from_numpy(image_data).to(device)
        tensor = tensor.unsqueeze(0) # Add channel dimension: (1, H, W)
    else:
        # RGB or RGBA image
        num_channels = info['planes']
        image_data = image_data.reshape(height, width, num_channels)
        
        # Keep only RGB if RGBA
        if num_channels == 4:
            image_data = image_data[:, :, :3]
            
        tensor = torch.from_numpy(image_data).to(device)
        # Permute from (H, W, C) to (C, H, W)
        tensor = tensor.permute(2, 0, 1)

    return tensor

def write_png_torch(tensor: torch.Tensor, path: str):
    """
    Writes a torch.Tensor to a PNG file.
    Handles RGB (8-bit) and Grayscale (16-bit) tensors.

    Args:
        tensor (torch.Tensor): The image tensor to save.
        path (str): The path to save the PNG file.
    """
    tensor = tensor.cpu()
    
    if tensor.dim() == 3 and tensor.shape in :
        # (C, H, W) format
        if tensor.shape == 3: # RGB
            # Permute from (C, H, W) to (H, W, C)
            tensor = tensor.permute(1, 2, 0)
            height, width, _ = tensor.shape
            # Reshape to (H, W*C) for PyPNG
            image_data = tensor.numpy().reshape(height, width * 3).astype(np.uint8)
            writer = png.Writer(width, height, greyscale=False, bitdepth=8)
        else: # Grayscale
            tensor = tensor.squeeze(0)
            height, width = tensor.shape
            image_data = tensor.numpy()
            bitdepth = 16 if image_data.dtype == np.uint16 or image_data.dtype == torch.int32 else 8
            if bitdepth == 16:
                image_data = image_data.astype(np.uint16)
            else:
                image_data = image_data.astype(np.uint8)
            writer = png.Writer(width, height, greyscale=True, bitdepth=bitdepth)
            
    elif tensor.dim() == 2: # Grayscale (H, W)
        height, width = tensor.shape
        image_data = tensor.numpy()
        bitdepth = 16 if image_data.dtype == np.uint16 or image_data.dtype == torch.int32 else 8
        if bitdepth == 16:
             image_data = image_data.astype(np.uint16)
        else:
             image_data = image_data.astype(np.uint8)
        writer = png.Writer(width, height, greyscale=True, bitdepth=bitdepth)
    else:
        raise ValueError("Unsupported tensor shape for PNG writing.")

    with open(path, 'wb') as f:
        writer.write(f, image_data)
