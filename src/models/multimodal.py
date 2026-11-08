import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision import FLAIRVisionEncoder
from models.tabular import FTTransformerEncoder

class BRSETMultimodalSSL(nn.Module):
    def __init__(self, embed_dim=256, num_tabular_features=5, temperature=0.07, vision_weights_path="../data/weights/flair_resnet.pth"):
        """
        The central Multimodal SSL framework tying Vision and Tabular branches.
        """
        super(BRSETMultimodalSSL, self).__init__()
        
        # 1. Initialize Encoders (pass the corrected relative path for weights)
        self.vision_encoder = FLAIRVisionEncoder(embed_dim=embed_dim, weights_path=vision_weights_path)
        self.tabular_encoder = FTTransformerEncoder(num_features=num_tabular_features, embed_dim=embed_dim)
        
        # 2. Learnable Temperature Parameter (Fixed .log() attribute error)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))
        
    def forward(self, images, tabular_data, masks):
        """
        Forward pass extracts embeddings from both modalities.
        """
        image_embeddings = self.vision_encoder(images)
        tabular_embeddings = self.tabular_encoder(tabular_data, masks)
        
        return image_embeddings, tabular_embeddings
    
    def compute_contrastive_loss(self, image_embeddings, tabular_embeddings):
        """
        Computes the Symmetric InfoNCE Loss.
        """
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        logits_per_image = logit_scale * image_embeddings @ tabular_embeddings.t()
        logits_per_tabular = logits_per_image.t()
        
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_tabular, labels)
        
        total_loss = (loss_i + loss_t) / 2
        
        return total_loss

# ==========================================
# Local testing block
# ==========================================
if __name__ == "__main__":
    batch_size = 4
    num_features = 5
    
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, num_features)
    dummy_masks = torch.ones(batch_size, num_features)
    
    model = BRSETMultimodalSSL(embed_dim=256, num_tabular_features=num_features)
    img_emb, tab_emb = model(dummy_images, dummy_tabular, dummy_masks)
    loss = model.compute_contrastive_loss(img_emb, tab_emb)
    
    print(f"Calculated InfoNCE Contrastive Loss: {loss.item():.4f}")