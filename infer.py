import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder_path_data = os.path.join(current_dir, 'data')
target_folder_path_models = os.path.join(current_dir, 'models')
target_folder_path_utils = os.path.join(current_dir, 'utils')

sys.path.insert(0, target_folder_path_data)
sys.path.insert(0, target_folder_path_models)
sys.path.insert(0, target_folder_path_utils)

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from dataset import ClearGraspViT_Dataset
from early_stopping import EarlyStopping
from vit_dense_prediction import create_vit_dense_predictor
import time

def get_full_paths(directory_path):
    """
    Returns a list containing the full paths of all files 
    within the specified directory and its subdirectories.
    """
    full_file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_file_paths.append(os.path.join(root, file))
    return full_file_paths

def train_encoders(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
      transforms.Resize((512, 512))
    ])

    # 1. Create Datasets and Dataloaders for the encoder training subset
    train_dataset = ClearGraspViT_Dataset(
        root_dir=config['data']['train_path'],
        subset='encoder',
        split_ratio=config['data']['split_ratio'],
        transform=train_transform,
        device=device
    )
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 2. Initialize Models
    normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3).to(device)
    # boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)
    # segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)

    normal_loss_fn = nn.MSELoss()

    # 4. Training Loop
    for file in get_full_paths('./checkpoints'):
        val_loss = 0
        normal_model.load_state_dict(torch.load(file, weights_only=True))
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)

                # Forward pass and loss for each model
                rgb = rgb / 255

                pred_normals = normal_model(rgb)
                loss_n = normal_loss_fn(torch.nn.functional.normalize(pred_normals, p=2, dim=1), batch['normals_gt'])
                
                loss = loss_n
                val_loss += loss.item()
    
        # Average validation loss
        val_loss /= len(val_loader)

        # Check early stopping condition
        early_stopping.check_early_stop(val_loss, normal_model)

        print(f"\Model  - ({file}) has validation loss: {val_loss}\n")

if __name__ == '__main__':
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_encoders(config)
