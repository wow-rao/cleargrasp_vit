#
# cleargrasp_vit/utils/torch_image_io.py
#
import torch
import png  # PyPNG library for PNG
from PIL import Image  # Pillow for JPG
import OpenEXR
import Imath
import numpy as np
from typing import Union

def read_png_torch(path: str, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Reads a PNG image file and returns it as a torch.Tensor."""
    reader = png.Reader(filename=path)
    width, height, rows, info = reader.read()
    
    image_data = np.vstack(list(map(np.uint8, rows)))
    
    if info['greyscale']:
        tensor = torch.from_numpy(image_data).to(device).unsqueeze(0)
    else:
        num_channels = info['planes']
        image_data = image_data.reshape(height, width, num_channels)
        if num_channels == 4:
            image_data = image_data[:, :, :3]
        tensor = torch.from_numpy(image_data).to(device).permute(2, 0, 1)
        
    return tensor

def read_jpg_torch(path: str, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """Reads a JPG image file using Pillow and returns it as a torch.Tensor."""
    img = Image.open(path)
    img_array = np.array(img)
    tensor = torch.from_numpy(img_array).to(device)
    # Permute from (H, W, C) to (C, H, W)
    tensor = tensor.permute(2, 0, 1)
    return tensor

def read_exr_torch(path: str, device: Union[str, torch.device] = 'cpu', test: bool = False) -> torch.Tensor:
    """Reads an EXR file (either single-channel depth or 3-channel normals) and returns a torch.Tensor."""
    exr_file = OpenEXR.InputFile(path)
    dw = exr_file.header()
    data_window = dw['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Determine pixel type and channels
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_str = list(dw['channels'].keys())

    if 'R' in channels_str and 'G' in channels_str and 'B' in channels_str:
        # 3-channel image (likely normals)
        r_str = exr_file.channel('R', pt)
        g_str = exr_file.channel('G', pt)
        b_str = exr_file.channel('B', pt)
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
        g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
        b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
        
        rgb = np.stack([r, g, b], axis=0)
        tensor = torch.from_numpy(rgb).to(device)
    else:
        # Assume single-channel image (likely depth)
        # Look for common depth channel names
        channel_name = 'Z'
        if channel_name not in channels_str:
            # Fallback to the first available channel if 'Z' is not present
            channel_name = channels_str
            
        channel_str = exr_file.channel(channel_name, pt)
        arr = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
        tensor = torch.from_numpy(arr).to(device).unsqueeze(0)
        
    return tensor
