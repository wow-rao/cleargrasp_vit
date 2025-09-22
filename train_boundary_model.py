#
# train_encoders.py (simplified)
#

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
from vit_dense_prediction import create_vit_dense_predictor
from torch_augmentations import Compose, Resize, Normalize, RandomHorizontalFlip
from early_stopping import EarlyStopping
import time

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
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 2. Initialize Models
    # normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3).to(device)
    boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)
    # segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)

    # 3. Initialize Optimizers and Loss Functions
    optimizer = torch.optim.AdamW(
        list(boundary_model.parameters()),
        lr=config['training']['learning_rate']
    )
    #normal_loss_fn = nn.MSELoss()
    boundary_loss_fn = nn.BCEWithLogitsLoss()
    #segmentation_loss_fn = nn.BCEWithLogitsLoss()

    # 4. Setup Early Stopping
    early_stopping = EarlyStopping(patience=1, verbose=True)

    start_time = time.time()

    # 4. Training Loop
    for epoch in range(config['training']['epochs']):
        train_loss = 0.0
        val_loss = 0.0

        #normal_model.train()
        boundary_model.train()
        #segmentation_model.train()
        for index, batch in enumerate(train_loader):
            rgb = batch['rgb'].to(device)
            #... get ground truths
            optimizer.zero_grad()
            
            # Forward pass and loss for each model
            rgb = rgb / 255

            #pred_normals = normal_model(rgb)
            #loss_n = normal_loss_fn(torch.nn.functional.normalize(pred_normals, p=2, dim=1), batch['normals_gt'])
            
            pred_boundaries = boundary_model(rgb)
            loss_b = boundary_loss_fn(torch.squeeze(pred_boundaries), torch.squeeze(batch['boundary_gt']))
            
            #pred_mask = segmentation_model(rgb)
            #loss_s = segmentation_loss_fn(torch.squeeze(pred_mask), torch.squeeze(batch['mask_gt']))
            
            total_loss = loss_b
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

            print(f"Batch {index + 1} / {len(train_loader)} completed in epoch {epoch + 1} with loss: {total_loss.item()}")
            print(f"ETA to Batch completion: {(len(train_loader) - index - 1) * ((time.time() - start_time) / (epoch * len(train_loader) + (index + 1))):.2f}s")
            print(f"ETA to Training completion: {((len(train_loader) - index - 1) + len(train_loader) * (config['training']['epochs'] - epoch - 1)) * ((time.time() - start_time) / (epoch * len(train_loader) + (index + 1))):.2f}s\n")

        #normal_model.eval()
        boundary_model.eval()
        #segmentation_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)

                # Forward pass and loss for each model
                rgb = rgb / 255

                #pred_normals = normal_model(rgb)
                #loss_n = normal_loss_fn(torch.nn.functional.normalize(pred_normals, p=2, dim=1), batch['normals_gt'])
            
                pred_boundaries = boundary_model(rgb)
                loss_b = boundary_loss_fn(torch.squeeze(pred_boundaries), torch.squeeze(batch['boundary_gt']))
            
                #pred_mask = segmentation_model(rgb)
                #loss_s = segmentation_loss_fn(torch.squeeze(pred_mask), torch.squeeze(batch['mask_gt']))
                
                loss = loss_b
                val_loss += loss.item()
    
        # Average validation loss
        val_loss /= len(val_loader) * val_loader.batch_size

        # Check early stopping condition
        early_stopping.check_early_stop(val_loss, boundary_model)

        print(f"\nEpoch {epoch + 1} completed with validation loss: {val_loss}\n")
    
        if early_stopping.stop_training:
            boundary_model.load_state_dict(early_stopping.weights)
            print(f"Early stopping at epoch {epoch}")
            break

        torch.save(boundary_model.state_dict(), f'./checkpoints/boundary_vit_epoch_{epoch + 1}.pth')
            
    # 5. Save model checkpoints
    torch.save(boundary_model.state_dict(), config['paths']['boundary_model_save'])
    #torch.save(boundary_model.state_dict(), config['paths']['boundary_model_save'])
    #torch.save(segmentation_model.state_dict(), config['paths']['segmentation_model_save'])

if __name__ == '__main__':
    with open('C:/Users/Donna/Downloads/cleargrasp_vit/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_encoders(config)
