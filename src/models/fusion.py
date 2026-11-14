import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CrossModalAttentionBlock(nn.Module):
    """
    Tokenizes 2048-D ResNet-50 features into 8 sub-tokens of 256-D and applies
    bi-directional Multi-Head Cross-Attention with SSL Image & Tabular embeddings.
    Also provides a Sample-Adaptive Instance Gate instead of a static scalar gate.
    """
    def __init__(self, embed_dim=256, num_vis_tokens=8, num_heads=4, dropout=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_vis_tokens = num_vis_tokens

        # Project 2048-D backbone features into 8 x 256-D visual sub-tokens
        self.vis_tokenizer = nn.Sequential(
            nn.Linear(2048, num_vis_tokens * embed_dim),
            nn.LayerNorm(num_vis_tokens * embed_dim),
            nn.GELU()
        )
        self.token_pos_emb = nn.Parameter(torch.randn(1, num_vis_tokens, embed_dim) * 0.02)

        # Tabular + SSL Image queries attend to high-res visual sub-tokens
        self.mm_to_vis_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Gated FFN on top of cross-attended tokens
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, img_emb, tab_emb, backbone_feat):
        B = img_emb.size(0)
        # Visual sub-tokens: (B, 8, 256)
        vis_tokens = self.vis_tokenizer(backbone_feat).view(B, self.num_vis_tokens, self.embed_dim)
        vis_tokens = self.norm_kv(vis_tokens + self.token_pos_emb)

        # Multimodal queries: [SSL_Image_Emb, SSL_Tab_Emb] -> (B, 2, 256)
        queries = self.norm_q(torch.stack([img_emb, tab_emb], dim=1))

        attn_out, _ = self.mm_to_vis_attn(query=queries, key=vis_tokens, value=vis_tokens)
        queries = queries + attn_out
        queries = self.norm_out(queries + self.ffn(queries))

        # Return enriched image and tabular embeddings + pooled visual context
        enriched_img = queries[:, 0, :]
        enriched_tab = queries[:, 1, :]
        vis_context = vis_tokens.mean(dim=1)
        return enriched_img, enriched_tab, vis_context


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
        self.fusion_type = str(self.fusion_config.get('fusion_type', 'concat_residual')).lower()
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

        if self.fusion_type == "cross_attention":
            self.cross_attn = CrossModalAttentionBlock(
                embed_dim=embed_dim, num_vis_tokens=8, num_heads=4, dropout=dropout_rate * 0.8
            )
            fusion_dim = embed_dim * 3  # enriched_img + enriched_tab + vis_context (768-D)

            self.multimodal_head = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim // 2, num_classes)
            )

            # Non-linear Bottleneck Visual Residual Head (replaces raw 2048->3 linear layer to stop Normal bias)
            self.visual_residual_head = nn.Sequential(
                nn.LayerNorm(2048),
                nn.Dropout(dropout_rate),
                nn.Linear(2048, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )

            # Sample-adaptive per-class gate (depends on patient features instead of static scalar)
            self.adaptive_gate = nn.Sequential(
                nn.Linear(fusion_dim, num_classes),
                nn.Sigmoid()
            )
        else:
            # Original V2 architecture (100% backward-compatible for finetune_dr3_v2.py)
            fusion_dim = embed_dim * 2
            self.multimodal_head = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
            self.visual_residual_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(2048, num_classes)
            )
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

    def forward(self, images, tabular_data, masks, return_features=False):
        img_emb, tab_emb, backbone_feat = self.ssl_base.forward_with_backbone(images, tabular_data, masks)
        new_img_emb, new_tab_emb = self.meta_fusion(img_emb, tab_emb)

        if self.fusion_type == "cross_attention" and backbone_feat is not None:
            enr_img, enr_tab, vis_ctx = self.cross_attn(new_img_emb, new_tab_emb, backbone_feat)
            fused_features = torch.cat((enr_img, enr_tab, vis_ctx), dim=1)
            mm_logits = self.multimodal_head(fused_features)
            vis_logits = self.visual_residual_head(backbone_feat)
            gate = self.adaptive_gate(fused_features)
            logits = (1.0 - gate) * vis_logits + gate * mm_logits * 1.5
            if return_features:
                return logits, fused_features
            return logits

        # Original V2 forward pass
        fused_features = torch.cat((new_img_emb, new_tab_emb), dim=1)
        mm_logits = self.multimodal_head(fused_features)

        if backbone_feat is not None:
            vis_logits = self.visual_residual_head(backbone_feat)
            gate = torch.sigmoid(self.multimodal_gate)
            logits = vis_logits + gate * mm_logits
        else:
            logits = mm_logits

        if return_features:
            return logits, fused_features
        return logits
