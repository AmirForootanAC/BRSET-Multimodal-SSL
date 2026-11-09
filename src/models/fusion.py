import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from models.multimodal import BRSETMultimodalSSL

class MetaFusionBlock(nn.Module):
    def __init__(self, dim):
        super(MetaFusionBlock, self).__init__()
        self.W_img_to_tab = nn.Linear(dim, dim)
        self.W_tab_to_img = nn.Linear(dim, dim)
        self.norm_img = nn.LayerNorm(dim)
        self.norm_tab = nn.LayerNorm(dim)

    def forward(self, img_emb, tab_emb):
        img_correction = img_emb * torch.tanh(img_emb * self.W_tab_to_img(tab_emb))
        tab_correction = tab_emb * torch.tanh(tab_emb * self.W_img_to_tab(img_emb))
        out_img = self.norm_img(img_emb + img_correction)
        out_tab = self.norm_tab(tab_emb + tab_correction)
        return out_img, out_tab

class BRSETClassifier(nn.Module):
    def __init__(self, config, task_name="dr_3class", vision_weights_path=None):
        super(BRSETClassifier, self).__init__()
        
        if 'task' in config:
            self.task_config = config['task']
        elif 'finetuning' in config and 'task' in config['finetuning']:
            self.task_config = config['finetuning']['task']
        else:
            self.task_config = {'num_classes': 3}
            
        self.fusion_config = config['fusion']
        embed_dim = int(config['model']['embed_dim'])
        num_classes = int(self.task_config.get('num_classes', 3))
        hidden_dim = int(self.fusion_config.get('hidden_dim', 256))
        dropout_rate = float(self.fusion_config.get('dropout_rate', 0.3))
        
        if vision_weights_path is None:
            vision_weights_path = config['model'].get('vision_weights', "../data/weights/flair_resnet.pth")
            
        self.ssl_base = BRSETMultimodalSSL(
            embed_dim=embed_dim, 
            num_tabular_features=int(config['dataset']['num_tabular_features']),
            temperature=float(config['model'].get('temperature', 0.07)),
            vision_weights_path=vision_weights_path
        )
        
        if self.fusion_config.get('freeze_ssl_backbone', False):
            for param in self.ssl_base.parameters():
                param.requires_grad = False
                
        self.meta_fusion = MetaFusionBlock(dim=embed_dim)
        fusion_dim = embed_dim * 2
        
        # شاخه پردازش تعاملی چندوجهی (Multimodal SSL Branch)
        self.multimodal_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # شاخه مستقیم ویژگی‌های 2048 بعدیِ تصویر (High-Resolution Visual Residual Stream)
        self.visual_residual_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, num_classes)
        )
        
        # ضریب یادگیرنده برای ترکیب هوشمند شاخه تصویری و شاخه چندوجهی
        self.multimodal_gate = nn.Parameter(torch.tensor(0.5))

    def load_ssl_weights(self, checkpoint_path, device):
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading pre-trained SSL weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.ssl_base.load_state_dict(state_dict, strict=False)
            print("✅ SSL weights successfully loaded into the backbone!")
        else:
            print(f"⚠️ SSL checkpoint not found at {checkpoint_path}. Proceeding with initialized weights.")

    def forward(self, images, tabular_data, masks):
        img_emb, tab_emb, backbone_feat = self.ssl_base.forward_with_backbone(images, tabular_data, masks)
        new_img_emb, new_tab_emb = self.meta_fusion(img_emb, tab_emb)
        fused_features = torch.cat((new_img_emb, new_tab_emb), dim=1)
        
        mm_logits = self.multimodal_head(fused_features)
        
        if backbone_feat is not None:
            vis_logits = self.visual_residual_head(backbone_feat)
            gate = torch.sigmoid(self.multimodal_gate)
            return vis_logits + gate * mm_logits
            
        return mm_logits