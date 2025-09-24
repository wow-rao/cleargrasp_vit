#
# cleargrasp_vit/models/vit_core.py
#
import torch
import torch.nn as nn
from native_sparse_attention_pytorch import SparseAttention

class PatchEmbedding(nn.Module):
  def __init__(self, in_channels, patch_size=4, dropout=0.001):
      super().__init__()
      embed_dim = (patch_size ** 2) * in_channels
      self.num_patches = (512 // patch_size) ** 2
      self.patch_size = patch_size
      self.patcher = nn.Sequential(
          # We use conv for doing the patching
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=embed_dim,
              # if kernel_size = stride -> no overlap
              kernel_size=patch_size,
              stride=patch_size
          ),
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
          nn.Flatten(2))
      self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
      self.position_embeddings = nn.Parameter(torch.randn(size=(1, self.num_patches+1, embed_dim)), requires_grad=True)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
      # Create a copy of the cls token for each of the elements of the BATCH
      cls_token = self.cls_token.expand(x.shape[0], -1, -1)
      # Create the patches
      x = self.patcher(x).permute(0, 2, 1)
      # Unify the position with the patches
      x = torch.cat([cls_token, x], dim=1)
      # Patch + Position Embedding
      x = self.position_embeddings + x
      x = self.dropout(x)
      return x
  
class ClassicDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layer1 = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0)
        self.layer2 = nn.BatchNorm2d(out_channels)
        self.layer3 = nn.ReLU(inplace=True)

            # Second deconvolution block
        self.layer4 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(6, 3),
                stride=(4 , 3),
                padding=0)

        # Final prediction layer as per Equation (2) in the paper.
        # A 1x1 convolution maps the features to the number of keypoints.
        self.predictor = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(torch.unsqueeze(x, 1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.predictor(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, expansion, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim*expansion)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim*expansion, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, image_dim, embed_dim, num_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.SAT = SparseAttention(image_dim, embed_dim, num_heads, 
                                   sliding_window_size = 2, compress_block_size = 4, 
                                   compress_block_sliding_stride = 2, selection_block_size = 4,
                                   num_selected_blocks = 2)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, expansion, dropout)
    
        
    def forward(self, x):
        norm1 = self.norm1(x)
        x = x + self.SAT(norm1)
        x = self.dropout(x)    
        norm2 = self.norm2(x)
        x = x + self.mlp(norm2)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, image_dim, embed_dim, num_heads, expansion, dropout, num_encoders):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(image_dim, embed_dim, num_heads, expansion, dropout) for _ in range(num_encoders)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, num_patches, dropout, in_channels, heads, depth, expansion, output_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(in_channels, num_patches, dropout)
        self.transformer_encoder =  TransformerEncoder(512, 128, num_heads, expansion, dropout, depth)
        self.decoder = ClassicDecoder(1, output_channels)
        
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x
