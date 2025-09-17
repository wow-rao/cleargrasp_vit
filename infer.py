#
# infer.py
#
import torch
import yaml
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cleargrasp_vit.utils.torch_image_io import read_png_torch, write_png_torch
from cleargrasp_vit.models.vit_dense_prediction import create_vit_dense_predictor
from cleargrasp_vit.models.depth_fusion_transformer import DepthFusionTransformer

def colorize_depth(depth_map):
    """Colorizes a depth map using a colormap for visualization."""
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    colored_map = plt.cm.viridis(depth_map_normalized.numpy())[:, :, :3]
    return (colored_map * 255).astype(np.uint8)

def infer(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Models
    print("Loading models...")
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
    normal_model.to(device).eval()
    boundary_model.to(device).eval()
    segmentation_model.to(device).eval()
    depth_fusion_model.to(device).eval()

    # 3. Load and Preprocess Inputs
    print(f"Reading inputs: {args.rgb_path}, {args.depth_path}")
    rgb_image = read_png_torch(args.rgb_path, device).float()
    depth_image = read_png_torch(args.depth_path, device).float()
    
    # Add batch dimension
    rgb_image = rgb_image.unsqueeze(0)
    depth_image = depth_image.unsqueeze(0)
    
    # TODO: Add normalization if models were trained with it

    # 4. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # Perception encoders
        mask_logits = segmentation_model(rgb_image)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float()
        
        pred_normals_raw = normal_model(rgb_image)
        pred_normals = torch.nn.functional.normalize(pred_normals_raw, p=2, dim=1)
        
        pred_boundaries_logits = boundary_model(rgb_image)
        pred_boundaries = torch.argmax(pred_boundaries_logits, dim=1, keepdim=True)
        
        # Depth fusion
        final_depth = depth_fusion_model(rgb_image, depth_image, pred_mask)

    # 5. Post-process and Save Outputs
    print(f"Saving results to {args.output_dir}")
    # Remove batch dimension and move to CPU
    final_depth = final_depth.squeeze().cpu()
    pred_mask = pred_mask.squeeze().cpu()
    pred_normals = pred_normals.squeeze().cpu()
    pred_boundaries = pred_boundaries.squeeze().cpu()

    # Save final depth (16-bit PNG, assuming depth is in mm)
    write_png_torch((final_depth * 1000).to(torch.int32), os.path.join(args.output_dir, "output_depth.png"))

    # Save visualizations
    # Colorized depth
    depth_viz = colorize_depth(final_depth)
    plt.imsave(os.path.join(args.output_dir, "output_depth_viz.png"), depth_viz)
    
    # Segmentation mask
    write_png_torch(pred_mask * 255, os.path.join(args.output_dir, "output_mask.png"))
    
    # Surface normals
    normals_viz = (pred_normals * 0.5 + 0.5) * 255 # Remap from [-1, 1] to 
    write_png_torch(normals_viz.to(torch.uint8), os.path.join(args.output_dir, "output_normals.png"))
    
    # Boundaries
    write_png_torch((pred_boundaries * 127).to(torch.uint8), os.path.join(args.output_dir, "output_boundaries.png"))
    
    print("Inference complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ClearGrasp-ViT inference on a single image pair.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--rgb_path', type=str, required=True, help='Path to the input RGB image.')
    parser.add_argument('--depth_path', type=str, required=True, help='Path to the input raw depth image.')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Directory to save the results.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    infer(args, config)
