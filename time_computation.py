import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder_path_models = os.path.join(current_dir, 'models')

sys.path.insert(0, target_folder_path_models)

import torch
import yaml
import numpy as np
from vit_dense_prediction import create_vit_dense_predictor
from vit_decoder import MultiImageVisionTransformer
import time

def resize_image_tensor(image_tensor, size):
    input_tensor_4d = image_tensor.unsqueeze(0)
    resized_tensor_4d = torch.nn.functional.interpolate(
        input_tensor_4d,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    resized_tensor_3d = resized_tensor_4d.squeeze(0)
    return resized_tensor_3d

def train_encoders(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Models
    normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3).to(device)
    boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)
    segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)

    channel_configs = [3, 3, 3, 3]  # RGB, Grayscale, 4-channel, RGB
    
    output_model = MultiImageVisionTransformer(
        img_size=512,
        patch_size=16,
        channel_configs=channel_configs,
        embed_dim=768,
        num_heads=12,
        hidden_dim=3072,
        dropout=0.1
    ).to(device)

    while True:
        normal_model.eval()
        boundary_model.eval()
        segmentation_model.eval()
        with torch.no_grad():
            rgb = np.load('./data/rgb.npy')
            rgb = torch.from_numpy(rgb).to(device) / 1
            rgb = resize_image_tensor(rgb, 512)
            rgb = torch.unsqueeze(rgb, 0)

            total_start = time.time()
            pred_normals = normal_model(rgb)
            print(f"The normal model takes {time.time() - total_start}s to run")
            start_time = time.time()
            pred_boundaries = boundary_model(rgb)
            print(f"The boundary model takes {time.time() - start_time}s to run")
            start_time = time.time()
            pred_mask = segmentation_model(rgb)
            print(f"The segmentation model takes {time.time() - start_time}s to run")
            start_time = time.time()
            pred_normals = model([rgb, rgb, rgb, rgb])
            print(f"The decoder model takes {time.time() - start_time}s to run")
            print(f"The entire model takes {time.time() - total_start}s to run")
    

if __name__ == '__main__':
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_encoders(config)
