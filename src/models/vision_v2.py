import os
import torch
import torch.nn as nn
from torchvision import models
from models.vision import FLAIRVisionEncoder

class DualVisionEncoder(nn.Module):
    def __init__(self, embed_dim=256, densenet_weight_path="../data/weights/densenet121.pth", flair_weight_path="../data/weights/flair_resnet.pth"):
        super(DualVisionEncoder, self).__init__()
        
        # Explicitly passing the correct path to preserve ophthalmology weights!
        self.flair = FLAIRVisionEncoder(weights_path=flair_weight_path)
        
        self.densenet = models.densenet121()
        
        if os.path.exists(densenet_weight_path):
            self.densenet.load_state_dict(torch.load(densenet_weight_path, map_location="cpu"), strict=False)
            print(f"DenseNet weights successfully loaded from {densenet_weight_path}")
        else:
            print("WARNING: DenseNet weights not found locally. Initialized randomly.")
            
        self.densenet.classifier = nn.Identity() 
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        f1 = self.flair(x)
        f2 = self.densenet(x)
        f_cat = torch.cat((f1, f2), dim=1)
        return self.fusion_proj(f_cat)