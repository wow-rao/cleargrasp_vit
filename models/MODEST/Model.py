"""
    The code partly borrows from
    https://github.com/antocad/FocusOnDepth
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision

from Reassemble import Reassemble
from Fusion import Fusion
from Head import HeadDepth, HeadSeg, MultiscaleHead

# torch.manual_seed(0)

class ISGNet(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 nclasses           = 1,
                 type               = "full",
                 model_timm         = "vit_large_patch16_384",
                 pretrain           = True,
                 iterations         = 3,
                 in_chans           = 3):
        """
        type : {"full", "depth", "seg"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        ## Transformer
        self.transformer_encoders_normal = timm.create_model(model_timm, pretrained=pretrain, in_chans=in_chans)
        self.transformer_encoders_seg = timm.create_model(model_timm, pretrained=pretrain, in_chans=in_chans)
        print("load vit successfully")
        self.type_ = type
        self.iterations = iterations

        ## Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        ## Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim, nclasses))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        ## Head
        if type == "full":
            self.head_multiscale = MultiscaleHead(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        elif type == "seg":
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        else:
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        
        self.depth_fc = [nn.Conv2d(1, 256, kernel_size=4, stride=16, padding=0, bias=True).to("cuda"), 
                         nn.Conv2d(1, 256, kernel_size=4, stride=16, padding=0, bias=True).to("cuda"), 
                         nn.Conv2d(1, 256, kernel_size=4, stride=16, padding=0, bias=True).to("cuda")]
            


    def forward(self, img, depth_img):

        _ = self.transformer_encoders_normal(img), self.transformer_encoders_seg(img)
        guide_depth, guide_seg, guide_normal = [], [], []
        out_depths, out_segs, out_normals = [], [], []

        ## multi-scale iterations
        for iter in range(self.iterations):
            depth_feature, seg_feature, normal_feature = self.depth_fc[iter](torchvision.transforms.Resize((192, 192))(depth_img)), None, None
            multiscale_depth, multiscale_seg, multiscale_normal = [], [], []
            depth_features, seg_features, normal_features = [], [], []
            for i in np.arange(len(self.fusions)-1, -1, -1):                        # 3, 2, 1, 0
                hook_to_take_normal = 'nt'+str(self.hooks[i])
                hook_to_take_seg = 'st'+str(self.hooks[i])
                activation_result_normal = self.activation[hook_to_take_normal]
                activation_result_seg = self.activation[hook_to_take_seg]
                reassemble_result_normal = self.reassembles[i](activation_result_normal)     
                reassemble_result_seg = self.reassembles[i](activation_result_seg)
                depth_feature, seg_feature, normal_feature = self.fusions[i](reassemble_result_normal, reassemble_result_seg, i, depth_feature, seg_feature,
                                                                normal_feature, guide_depth, guide_seg, guide_normal)     # [256, 24, 24], [256, 48, 48], [256, 96, 96], [256, 192, 192]      
                output_depth, output_seg, output_normal = self.head_multiscale(depth_feature, seg_feature, normal_feature)     # [256, 48, 48], [256, 96, 96], [256, 192, 192], [256, 384, 384]
                multiscale_depth.append(output_depth)
                multiscale_seg.append(output_seg)
                multiscale_normal.append(output_normal)
                depth_features.append(depth_feature)
                seg_features.append(seg_feature)
                normal_features.append(normal_feature)
            guide_depth.append(depth_features)
            guide_seg.append(seg_features)
            guide_normal.append(normal_features)
            out_depths.append(multiscale_depth)
            out_segs.append(multiscale_seg)
            out_normals.append(multiscale_normal)
        
        return out_depths, out_segs, out_normals


    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders_normal.blocks[h].register_forward_hook(get_activation('nt'+str(h)))
            self.transformer_encoders_seg.blocks[h].register_forward_hook(get_activation('st'+str(h)))