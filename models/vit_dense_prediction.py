#
# cleargrasp_vit/models/vit_dense_prediction.py
#
import torch
import torch.nn as nn
from vit_core import VisionTransformer

class ConvUpsampler(nn.Module):
    """
    A simple convolutional upsampling block.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(self.upsample(x))

class ViTDensePrediction(nn.Module):
    """
    Vision Transformer for dense prediction tasks like segmentation or normal estimation.

    Args:
        vit_model (VisionTransformer): The pre-configured ViT encoder.
        decoder_channels (list[int]): List of channel numbers for the upsampling blocks.
        output_channels (int): Number of output channels for the final prediction.
    """
    def __init__(self, vit_model: VisionTransformer, decoder_channels: list, output_channels: int):
        super().__init__()
        self.vit = vit_model
        self.patch_size = self.vit.patch_embedding.patch_size
        self.embed_dim = self.vit.patch_embedding.projection.out_channels
        
        # Build the decoder
        in_ch = self.embed_dim
        decoder_blocks = []
        for out_ch in decoder_channels:
            decoder_blocks.append(ConvUpsampler(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.Sequential(*decoder_blocks)
        
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Get patch embeddings from the ViT encoder
        hidden_state = self.vit.get_last_hidden_state(x) # Shape: (B, num_patches, dim)
        
        # Reshape sequence back to a 2D feature map
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Reshape: (B, H/p * W/p, dim) -> (B, dim, H/p, W/p)
        features = hidden_state.transpose(1, 2).reshape(B, self.embed_dim, num_patches_h, num_patches_w)
        
        # Upsample features with the decoder
        decoded_features = self.decoder(features)
        
        # Final 1x1 convolution to get the output map
        output = self.final_conv(decoded_features)
        
        # Upsample to original image size
        output = nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output

def create_vit_dense_predictor(model_config: dict, output_channels: int) -> ViTDensePrediction:
    """Helper function to create a ViT dense prediction model from a config dict."""
    vit_encoder = VisionTransformer(**model_config)
    
    # Example decoder config, can be customized
    decoder_config = [model_config['dim'] // 2, model_config['dim'] // 4, model_config['dim'] // 8]
    
    model = ViTDensePrediction(
        vit_model=vit_encoder,
        decoder_channels=decoder_config,
        output_channels=output_channels
    )
    return model
