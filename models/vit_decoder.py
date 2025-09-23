import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them with positional encoding"""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding projection
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        
        # Create patches and project to embedding dimension
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding that includes both spatial position and image index"""
    
    def __init__(self, embed_dim: int, max_patches: int, max_images: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Spatial positional encoding
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim // 2))
        
        # Image index encoding
        self.image_pos_embed = nn.Parameter(torch.randn(1, max_images, embed_dim // 2))
        
    def forward(self, x, image_idx: int):
        # x: (batch_size, n_patches, embed_dim)
        B, n_patches, _ = x.shape
        
        # Get spatial positional encoding
        spatial_pos = self.spatial_pos_embed[:, :n_patches, :]  # (1, n_patches, embed_dim//2)
        
        # Get image positional encoding
        image_pos = self.image_pos_embed[:, image_idx:image_idx+1, :]  # (1, 1, embed_dim//2)
        image_pos = image_pos.expand(-1, n_patches, -1)  # (1, n_patches, embed_dim//2)
        
        # Concatenate spatial and image positional encodings
        pos_encoding = torch.cat([spatial_pos, image_pos], dim=-1)  # (1, n_patches, embed_dim)
        
        return x + pos_encoding

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class ConvBlock(nn.Module):
    """Convolutional block for processing attention output"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, embed_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch_size, n_patches, embed_dim)
        residual = x
        
        # Layer norm and attention
        x = self.norm1(x)
        
        # Convolution (transpose for conv1d)
        x = x.transpose(1, 2)  # (B, embed_dim, n_patches)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Residual connection
        x = residual + self.dropout(x)
        x = self.norm2(x)
        
        return x

class TransformerBlock(nn.Module):
    """Complete transformer block with attention and convolution"""
    
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.conv_block = ConvBlock(embed_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm(x)
        x = self.attention(x)
        x = residual + x
        
        # Convolutional processing
        x = self.conv_block(x)
        
        return x

class MultiImageVisionTransformer(nn.Module):
    """Multi-image Vision Transformer that processes n images sequentially"""
    
    def __init__(
        self, 
        img_size: int = 224,
        patch_size: int = 16,
        channel_configs: List[int] = [3, 3, 3, 3],  # Channels for each image
        embed_dim: int = 768,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_images = len(channel_configs)
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embeddings for each image (different channel counts)
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(img_size, patch_size, channels, embed_dim)
            for channels in channel_configs
        ])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, self.n_patches, self.num_images)
        
        # Transformer blocks (one less than number of images since we start with 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(self.num_images - 1)
        ])
        
        # Image reconstruction head
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * 1)  # Single channel output
        )
        
        # Store reconstruction parameters
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, images: List[torch.Tensor]):
        """
        Forward pass through the multi-image transformer
        
        Args:
            images: List of tensors, each of shape (batch_size, channels, height, width)
        """
        assert len(images) == self.num_images, f"Expected {self.num_images} images, got {len(images)}"
        
        # Process first two images
        x1 = self.patch_embeddings[0](images[0])  # (B, n_patches, embed_dim)
        x2 = self.patch_embeddings[1](images[1])  # (B, n_patches, embed_dim)
        
        # Add positional encoding
        x1 = self.pos_encoding(x1, 0)
        x2 = self.pos_encoding(x2, 1)
        
        # Concatenate patches from both images
        x = torch.cat([x1, x2], dim=1)  # (B, 2*n_patches, embed_dim)
        x = self.dropout(x)
        
        # Process through first transformer block
        x = self.transformer_blocks[0](x)
        
        # Process remaining images sequentially
        for i in range(2, self.num_images):
            # Get next image patches
            x_new = self.patch_embeddings[i](images[i])
            x_new = self.pos_encoding(x_new, i)
            
            # Concatenate with previous output
            x = torch.cat([x, x_new], dim=1)  # (B, (i+1)*n_patches, embed_dim)
            x = self.dropout(x)
            
            # Process through transformer block
            x = self.transformer_blocks[i-1](x)
        
        # Global average pooling and image reconstruction
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Reconstruct image patches
        patches = self.reconstruction_head(x)  # (B, patch_size^2 * 1)
        
        # Reshape to image format
        batch_size = patches.shape[0]
        patches_per_side = self.img_size // self.patch_size
        
        # Reshape patches to (B, 1, patch_size, patch_size, patches_per_side, patches_per_side)
        patches = patches.view(batch_size, 1, self.patch_size, self.patch_size, 1, 1)
        patches = patches.expand(-1, -1, -1, -1, patches_per_side, patches_per_side)
        
        # Rearrange to form complete image
        # This is a simplified reconstruction - you might want to use a more sophisticated decoder
        output_img = patches.permute(0, 1, 4, 2, 5, 3).contiguous()
        output_img = output_img.view(batch_size, 1, self.img_size, self.img_size)
        
        return output_img