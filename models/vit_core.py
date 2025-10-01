#
# cleargrasp_vit/models/vit_core.py
#
import torch
import torch.nn as nn

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

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, num_patches, dropout, in_channels, heads, depth, expansion, output_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(in_channels, num_patches, dropout)
        self.blocks = []
        for i in range(depth):
            self.blocks.append(nn.TransformerEncoderLayer(d_model=(patch_size ** 2) * in_channels, 
                                                      nhead=heads, dropout=dropout, 
                                                      dim_feedforward=int(((patch_size ** 2) * in_channels)*expansion), 
                                                      activation="gelu", batch_first=True, norm_first=True))
        self.blocks = nn.ModuleList(self.blocks)
        self.decoder_block = ClassicDecoder(1, output_channels)
        
    def forward(self, x):
        x = self.embeddings_block(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return self.decoder_block(x)
