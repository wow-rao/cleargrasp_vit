#
# cleargrasp_vit/models/depth_fusion_transformer.py
#
import torch
import torch.nn as nn
from.vit_core import PatchEmbedding, TransformerBlock

class CrossAttention(nn.Module):
    """
    Cross-Attention module. The query comes from one source (e.g., depth),
    and the key/value come from another (e.g., semantic features).
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        # x_query shape: (B, N_q, D)
        # x_context shape: (B, N_c, D)
        B, N_q, D = x_query.shape
        _, N_c, _ = x_context.shape
        
        q = self.to_q(x_query).reshape(B, N_q, self.heads, D // self.heads).transpose(1, 2)
        kv = self.to_kv(x_context).reshape(B, N_c, 2, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv, kv # Shape: (B, heads, N_c, head_dim)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.to_out(out)

class CrossAttentionBlock(nn.Module):
    """
    A block containing self-attention on the query, followed by cross-attention
    with the context, and an MLP.
    """
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, heads, dropout) # From vit_core.py
        
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, dropout)
        
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_dim, dropout) # From vit_core.py

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        # Self-attention on query tokens
        x_query = x_query + self.self_attn(self.norm_q1(x_query))
        # Cross-attention with context tokens
        x_query = x_query + self.cross_attn(self.norm_q2(x_query), self.norm_c(x_context))
        # MLP
        x_query = x_query + self.mlp(self.norm_mlp(x_query))
        return x_query

class DepthFusionTransformer(nn.Module):
    """
    Fuses latent features and a modified depth map to reconstruct the final depth.

    Args:
        normal_estimator (ViTDensePrediction): Trained normal estimation model.
        boundary_detector (ViTDensePrediction): Trained boundary detection model.
        vit_config (dict): Configuration for the internal ViT components.
        depth_decoder_channels (list[int]): Decoder channels for upsampling.
    """
    def __init__(self, normal_estimator: ViTDensePrediction, boundary_detector: ViTDensePrediction,
                 vit_config: dict, depth_decoder_channels: list):
        super().__init__()
        # Freeze the pre-trained encoders
        self.normal_estimator = normal_estimator
        self.boundary_detector = boundary_detector
        for param in self.normal_estimator.parameters():
            param.requires_grad = False
        for param in self.boundary_detector.parameters():
            param.requires_grad = False
            
        dim = vit_config['dim']
        patch_size = vit_config['patch_size']
        
        # Patch embedding for the modified depth input (RGB + Masked Depth = 4 channels)
        self.depth_patch_embedding = PatchEmbedding(
            vit_config['image_size'], patch_size, in_channels=4, embed_dim=dim
        )
        num_patches = self.depth_patch_embedding.num_patches
        self.depth_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # Cross-attention decoder blocks
        self.decoder_blocks = nn.ModuleList(, vit_config['mlp_dim'], vit_config['dropout'])
            for _ in range(vit_config['depth'])
        ])
        
        # Final upsampling head to reconstruct the depth image
        in_ch = dim
        upsampler_blocks =
        for out_ch in depth_decoder_channels:
            upsampler_blocks.append(ConvUpsampler(in_ch, out_ch, scale_factor=2))
            in_ch = out_ch
        self.upsampler_head = nn.Sequential(*upsampler_blocks)
        self.final_depth_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, rgb_image: torch.Tensor, depth_image: torch.Tensor, transparent_mask: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb_image.shape
        
        # 1. Get latent features from frozen encoders
        with torch.no_grad():
            self.normal_estimator.eval()
            self.boundary_detector.eval()
            # Get the feature maps before the final upsampling
            normal_latent = self.normal_estimator.vit.get_last_hidden_state(rgb_image)
            boundary_latent = self.boundary_detector.vit.get_last_hidden_state(rgb_image)
        
        # Concatenate latent features to form the context
        context_tokens = torch.cat([normal_latent, boundary_latent], dim=1) # Shape: (B, 2*N, D)
        
        # 2. Prepare the query input
        # Mask the input depth where transparent objects are present
        modified_depth = depth_image * (1 - transparent_mask)
        # Concatenate with RGB as input to the query patch embedding
        query_input = torch.cat([rgb_image, modified_depth], dim=1) # Shape: (B, 4, H, W)
        
        query_tokens = self.depth_patch_embedding(query_input)
        query_tokens += self.depth_pos_embedding
        
        # 3. Pass through the cross-attention decoder
        for block in self.decoder_blocks:
            query_tokens = block(query_tokens, context_tokens)
            
        # 4. Reconstruct the final depth map
        num_patches_h = H // self.depth_patch_embedding.patch_size
        num_patches_w = W // self.depth_patch_embedding.patch_size
        
        # Reshape: (B, N_q, D) -> (B, D, H/p, W/p)
        decoder_output_features = query_tokens.transpose(1, 2).reshape(B, -1, num_patches_h, num_patches_w)
        
        upsampled_features = self.upsampler_head(decoder_output_features)
        final_depth = self.final_depth_conv(upsampled_features)
        final_depth = nn.functional.interpolate(final_depth, size=(H, W), mode='bilinear', align_corners=False)
        
        return final_depth
