#
# train_encoders.py (simplified)
#
import torch
import yaml
from torch.utils.data import DataLoader
from cleargrasp_vit.data.dataset import ClearGraspViT_Dataset
from cleargrasp_vit.models.vit_dense_prediction import create_vit_dense_predictor

def train_encoders(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create Datasets and Dataloaders for the encoder training subset
    train_dataset = ClearGraspViT_Dataset(
        root_dir=config['data']['train_path'],
        subset='encoder',
        split_ratio=config['data']['split_ratio'],
        #... add transforms from config
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # 2. Initialize Models
    normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3).to(device)
    boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3).to(device)
    segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1).to(device)

    # 3. Initialize Optimizers and Loss Functions
    optimizer = torch.optim.AdamW(
        list(normal_model.parameters()) + list(boundary_model.parameters()) + list(segmentation_model.parameters()),
        lr=config['training']['learning_rate']
    )
    normal_loss_fn = nn.L1Loss()
    boundary_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 5.0]).to(device)) # Weighted loss
    segmentation_loss_fn = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    for epoch in range(config['training']['epochs']):
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            #... get ground truths
            
            optimizer.zero_grad()
            
            # Forward pass and loss for each model
            pred_normals = normal_model(rgb)
            loss_n = normal_loss_fn(torch.nn.functional.normalize(pred_normals, p=2, dim=1), batch['normals_gt'])
            
            pred_boundaries = boundary_model(rgb)
            loss_b = boundary_loss_fn(pred_boundaries, batch['boundary_gt'])
            
            pred_mask = segmentation_model(rgb)
            loss_s = segmentation_loss_fn(pred_mask, batch['mask_gt'])
            
            total_loss = loss_n + loss_b + loss_s
            total_loss.backward()
            optimizer.step()
            
            #... logging
            
    # 5. Save model checkpoints
    torch.save(normal_model.state_dict(), config['paths']['normal_model_save'])
    #... save other models

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_encoders(config)
