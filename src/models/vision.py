import os
import torch
import torch.nn as nn
from torchvision import models

class FLAIRVisionEncoder(nn.Module):
    def __init__(self, embed_dim=256, weights_path="data/weights/flair_resnet.pth"):
        """
        Image encoder based on FLAIR architecture using local weights
        """
        super(FLAIRVisionEncoder, self).__init__()
        
        # 1. Build the raw backbone
        resnet = models.resnet50(weights=None)
        
        # Check if weights file exists and load it
        if os.path.exists(weights_path):
            print(f"[INFO] Loading local FLAIR weights from: {weights_path}")
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                
                # Fix additional prefixes in GitHub checkpoint files
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('backbone.', '').replace('module.', '').replace('resnet.', '')
                    new_state_dict[new_key] = v
                    
                # strict=False to ignore the final classification layer
                resnet.load_state_dict(new_state_dict, strict=False)
                print("[SUCCESS] Ophthalmology-specific weights successfully loaded into ResNet-50!")
            except Exception as e:
                print(f"[ERROR] Failed to load weights file: {e}")
        else:
            print(f"[WARNING] File {weights_path} not found!")
            print("Using default ImageNet weights instead...")
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
        # 2. Remove the final layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_dim = resnet.fc.in_features
        
        # 3. Build Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.BatchNorm1d(self.backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_dim, embed_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.projection_head(features)
        
        # 4. L2 Normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# ==========================================
# Run local encoder test
# ==========================================
if __name__ == "__main__":
    dummy_images = torch.randn(4, 3, 224, 224)
    
    # The model reads the default path 'data/weights/flair_resnet.pth' internally
    model = FLAIRVisionEncoder(embed_dim=256)
    
    output_embeddings = model(dummy_images)
    print("Image output embedding shape (V):", output_embeddings.shape)