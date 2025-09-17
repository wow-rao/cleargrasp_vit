#
# train_decoder.py (simplified)
#
import torch
import yaml
from torch.utils.data import DataLoader
from cleargrasp_vit.data.dataset import ClearGraspViT_Dataset
from cleargrasp_vit.models.vit_dense_prediction import create_vit_dense_predictor
from cleargrasp_vit.models.depth_fusion_transformer import DepthFusionTransformer

def train_decoder(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create Dataset for the decoder training subset
    train_dataset = ClearGraspViT_Dataset(
        root_dir=config['data']['train_path'],
        subset='decoder',
        split_ratio=config['data']['split_ratio'],
        #... add transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size_decoder'], shuffle=True)

    # 2. Load pre-trained encoder models
    normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3)
    normal_model.load_state_dict(torch.load(config['paths']['normal_model_save']))
    boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3)
    boundary_model.load_state_dict(torch.load(config['paths']['boundary_model_save']))
    segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1) # Needed for mask
    segmentation_model.load_state_dict(torch.load(config['paths']['segmentation_model_save']))
    
    # 3. Initialize the Depth Fusion Transformer
    depth_fusion_model = DepthFusionTransformer(
        normal_estimator=normal_model,
        boundary_detector=boundary_model,
        vit_config=config['model']['vit'],
        depth_decoder_channels=config['model']['depth_decoder_channels']
    ).to(device)

    # 4. Initialize Optimizer and Loss Function
    optimizer = torch.optim.AdamW(depth_fusion_model.parameters(), lr=config['training']['learning_rate_decoder'])
    depth_loss_fn = nn.L1Loss()

    # 5. Training Loop
    for epoch in range(config['training']['epochs_decoder']):
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            
            # Infer the transparent mask with the frozen segmentation model
            with torch.no_grad():
                mask_logits = segmentation_model(rgb)
                transparent_mask = (torch.sigmoid(mask_logits) > 0.5).float()

            pred_depth = depth_fusion_model(rgb, depth, transparent_mask)
            
            # Calculate loss only on transparent regions
            loss = depth_loss_fn(pred_depth[transparent_mask.bool()], batch['depth_gt'][transparent_mask.bool()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #... logging
            
    # 6. Save the final model
    torch.save(depth_fusion_model.state_dict(), config['paths']['depth_fusion_model_save'])

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_decoder(config)
