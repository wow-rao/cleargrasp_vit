#
# evaluate.py
#
import torch
import yaml
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from cleargrasp_vit.data.dataset import ClearGraspViT_Dataset
from cleargrasp_vit.models.vit_dense_prediction import create_vit_dense_predictor
from cleargrasp_vit.models.depth_fusion_transformer import DepthFusionTransformer

def compute_depth_metrics(pred, gt, mask):
    """Computes depth metrics on the masked regions."""
    pred_m = pred[mask]
    gt_m = gt[mask]
    
    if gt_m.numel() == 0:
        return None

    # RMSE
    rmse = torch.sqrt(torch.mean((pred_m - gt_m) ** 2))
    
    # Relative Error
    rel = torch.mean(torch.abs(pred_m - gt_m) / gt_m)
    
    # Delta thresholds
    thresh = torch.max((gt_m / pred_m), (pred_m / gt_m))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    
    return {
        'rmse': rmse.item(),
        'rel': rel.item(),
        'd1': d1.item(),
        'd2': d2.item(),
        'd3': d3.item()
    }

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Models
    normal_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3)
    normal_model.load_state_dict(torch.load(config['paths']['normal_model_save'], map_location=device))
    
    boundary_model = create_vit_dense_predictor(config['model']['vit'], output_channels=3)
    boundary_model.load_state_dict(torch.load(config['paths']['boundary_model_save'], map_location=device))
    
    segmentation_model = create_vit_dense_predictor(config['model']['vit'], output_channels=1)
    segmentation_model.load_state_dict(torch.load(config['paths']['segmentation_model_save'], map_location=device))
    
    depth_fusion_model = DepthFusionTransformer(
        normal_estimator=normal_model,
        boundary_detector=boundary_model,
        vit_config=config['model']['vit'],
        depth_decoder_channels=config['model']['depth_decoder_channels']
    )
    depth_fusion_model.load_state_dict(torch.load(config['paths']['depth_fusion_model_save'], map_location=device))
    
    # Set models to evaluation mode
    segmentation_model.to(device).eval()
    depth_fusion_model.to(device).eval()

    # 2. Create Test Dataset
    test_dataset = ClearGraspViT_Dataset(
        root_dir=config['data']['test_path'],
        subset='all',
        device=device
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. Evaluation Loop
    results =
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            rgb = batch['rgb']
            depth = batch['depth']
            depth_gt = batch['depth_gt']
            mask_gt = batch['mask_gt'].bool()

            # Full inference pipeline
            mask_logits = segmentation_model(rgb)
            transparent_mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()
            
            pred_depth = depth_fusion_model(rgb, depth, transparent_mask_pred)
            
            # Compute metrics only on valid ground truth mask pixels
            metrics = compute_depth_metrics(pred_depth, depth_gt, mask_gt)
            if metrics:
                metrics['sample_name'] = batch['name']
                results.append(metrics)

    # 4. Save and Print Results
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/evaluation_metrics.csv', index=False)
    
    print("\n--- Evaluation Summary ---")
    print(results_df.mean(numeric_only=True))
    print("\nResults saved to results/evaluation_metrics.csv")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    evaluate(config)
