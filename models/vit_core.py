#
# cleargrasp_vit/models/vit_core.py
#
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Converts a 2D image into a sequence of 1D patch embeddings.

    Args:
        image_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each patch (assumed square).
        in_channels (int): Number of input channels.
        embed_dim (int): The embedding dimension for each patch.
    """
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # A convolution layer is an efficient way to implement patch embedding.
        # kernel_size and stride are set to patch_size to create non-overlapping patches.
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, C, H, W)
        x = self.projection(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)       # Shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        dim (int): Total dimension of the model.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, N, D), where N is num_patches, D is dim
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv, qkv, qkv  # Shape: (B, heads, N, head_dim)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)

class MLPBlock(nn.Module):
    """
    Feed-forward network (MLP) block.
    """
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single block of the Vision Transformer encoder.
    """
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    The complete Vision Transformer model.

    Args:
        image_size (int): Size of the input image.
        patch_size (int): Size of each patch.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the MLP block.
        dropout (float): Dropout rate.
        emb_dropout (float): Dropout rate for embeddings.
    """
    def __init__(self, *, image_size: int, patch_size: int, in_channels: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.0, emb_dropout: float = 0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embedding.num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_blocks = nn.ModuleList()

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def get_last_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        B, N, _ = x.shape
        x += self.pos_embedding[:, :N]
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)
        
        return self.to_latent(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This forward is for classification. For dense prediction, we'll use a different head.
        x = self.get_last_hidden_state(x)
        # For a simple classification head, one might average the patch tokens.
        x = x.mean(dim=1)
        return self.mlp_head(x)
