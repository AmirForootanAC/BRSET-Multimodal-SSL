import torch
import torch.nn as nn
from models.multimodal import BRSETMultimodalSSL
from models.vision_v2 import DualVisionEncoder

class BRSETMultimodalSSL_V2(BRSETMultimodalSSL):
    def __init__(self, embed_dim=256, num_tabular_features=6, temperature=0.1):
        super(BRSETMultimodalSSL_V2, self).__init__(
            embed_dim=embed_dim, 
            num_tabular_features=num_tabular_features, 
            temperature=temperature
        )
        
        # The new DualVisionEncoder seamlessly outputs embed_dim (256)
        self.vision_encoder = DualVisionEncoder(embed_dim=embed_dim)