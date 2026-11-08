import torch
import torch.nn as nn
from models.multimodal import BRSETMultimodalSSL

class MetaFusionBlock(nn.Module):
    def __init__(self, dim):
        super(MetaFusionBlock, self).__init__()
        self.W_img_to_tab = nn.Linear(dim, dim)
        self.W_tab_to_img = nn.Linear(dim, dim)

    def forward(self, img_emb, tab_emb):
        img_correction = img_emb * torch.tanh(img_emb * self.W_tab_to_img(tab_emb))
        tab_correction = tab_emb * torch.tanh(tab_emb * self.W_img_to_tab(img_emb))

        new_img_emb = img_emb + img_correction
        new_tab_emb = tab_emb + tab_correction

        return new_img_emb, new_tab_emb

class BRSETClassifier(nn.Module):
    def __init__(self, config, task_name="multilabel_5"):
        super(BRSETClassifier, self).__init__()
        
        self.task_config = config['finetuning']['tasks'][task_name]
        self.fusion_config = config['fusion']
        
        embed_dim = config['model']['embed_dim']
        num_classes = self.task_config['num_classes']
        hidden_dim = self.fusion_config['hidden_dim']
        dropout_rate = self.fusion_config['dropout_rate']
        
        self.ssl_base = BRSETMultimodalSSL(
            embed_dim=embed_dim, 
            num_tabular_features=config['dataset']['num_tabular_features']
        )
        
        if self.fusion_config.get('freeze_ssl_backbone', False):
            for param in self.ssl_base.parameters():
                param.requires_grad = False
                
        self.meta_fusion = MetaFusionBlock(dim=embed_dim)
        
        fusion_dim = embed_dim * 2 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def load_ssl_weights(self, checkpoint_path, device):
        print(f"Loading pre-trained SSL weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.ssl_base.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("SSL weights successfully loaded into the backbone!")

    def forward(self, images, tabular_data, masks):
        img_emb, tab_emb = self.ssl_base(images, tabular_data, masks)
        
        new_img_emb, new_tab_emb = self.meta_fusion(img_emb, tab_emb)
        
        fused_features = torch.cat((new_img_emb, new_tab_emb), dim=1)
        
        logits = self.classifier(fused_features)
        return logits