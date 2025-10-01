import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from vit_dense_prediction import create_vit_dense_predictor

class Encoder(nn.Module):
    """
    Converts a feature vector (B, 1000) to an image (B, 1, 384, 384)
    Uses progressive upsampling with transposed convolutions
    """
    
    def __init__(self, config, vector_dim=1000, img_size=384, out_channels=1):
        super().__init__()
        
        self.model = create_vit_dense_predictor(config, output_channels=3)
        self.img_size = img_size
        
        # Calculate initial spatial size
        # We'll start at 6x6 and upsample to 384x384
        # 6 -> 12 -> 24 -> 48 -> 96 -> 192 -> 384 (6 upsampling stages, 2x each)
        self.init_size = 6
        self.init_channels = 512
        
        # FC layer to reshape vector to initial spatial feature map
        self.fc = nn.Linear(vector_dim, self.init_channels * self.init_size * self.init_size)
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            # 6x6 -> 12x12
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 12x12 -> 24x24
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 24x24 -> 48x48
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 48x48 -> 96x96
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 96x96 -> 192x192
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 192x192 -> 384x384
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final convolution to get desired output channels
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # or Tanh() depending on your output range
        )
        
    def forward(self, image):
        vector = self.model(image)
        B = vector.shape[0]
        
        # Project to spatial feature map
        x = self.fc(vector)  # (B, 512*6*6)
        x = x.view(B, self.init_channels, self.init_size, self.init_size)  # (B, 512, 6, 6)
        
        # Upsample through decoder
        x = self.decoder(x)  # (B, 1, 384, 384)
        

        return x








