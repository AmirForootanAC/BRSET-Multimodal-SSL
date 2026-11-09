import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision import FLAIRVisionEncoder
from models.tabular import FTTransformerEncoder

class BRSETMultimodalSSL(nn.Module):
    def __init__(self, embed_dim=256, num_tabular_features=8, temperature=0.07, vision_weights_path="../data/weights/flair_resnet.pth"):
        super(BRSETMultimodalSSL, self).__init__()
        
        self.vision_encoder = FLAIRVisionEncoder(embed_dim=embed_dim, weights_path=vision_weights_path)
        self.tabular_encoder = FTTransformerEncoder(num_features=num_tabular_features, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))
        
        # ذخیره ویژگی‌های 2048 بعدیِ خام ResNet قبل از لایه فشرده‌ساز
        self.latest_backbone_features = None
        self._register_backbone_hook()

    def _register_backbone_hook(self):
        for module in self.vision_encoder.modules():
            if isinstance(module, nn.Linear) and module.in_features == 2048:
                def capture_input_hook(mod, inputs, output):
                    self.latest_backbone_features = inputs[0]
                module.register_forward_hook(capture_input_hook)
                break

    def forward(self, images, tabular_data, masks):
        image_embeddings = self.vision_encoder(images)
        tabular_embeddings = self.tabular_encoder(tabular_data, masks)
        return image_embeddings, tabular_embeddings

    def forward_with_backbone(self, images, tabular_data, masks):
        image_embeddings = self.vision_encoder(images)
        tabular_embeddings = self.tabular_encoder(tabular_data, masks)
        backbone_feat = self.latest_backbone_features
        return image_embeddings, tabular_embeddings, backbone_feat

    def compute_contrastive_loss(self, image_embeddings, tabular_embeddings, labels=None, supcon_weight=0.4):
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        
        # 1. Symmetric Multimodal InfoNCE Loss (Image <-> Tabular)
        logits_per_image = logit_scale * (image_embeddings @ tabular_embeddings.t())
        logits_per_tabular = logits_per_image.t()
        
        batch_size = image_embeddings.shape[0]
        diag_labels = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
        
        loss_i = F.cross_entropy(logits_per_image, diag_labels)
        loss_t = F.cross_entropy(logits_per_tabular, diag_labels)
        infonce_loss = 0.5 * (loss_i + loss_t)
        
        if labels is None or supcon_weight <= 0.0:
            return infonce_loss, infonce_loss.detach(), torch.tensor(0.0, device=image_embeddings.device)

        # 2. Class-Aware Multimodal Supervised Contrastive Loss
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(image_embeddings.device)
        
        # پایدارسازی عددی برای Log-Softmax
        logits_max, _ = torch.max(logits_per_image, dim=1, keepdim=True)
        logits_stable = logits_per_image - logits_max.detach()
        exp_logits = torch.exp(logits_stable)
        log_prob = logits_stable - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1.0)
        supcon_loss = -mean_log_prob_pos.mean()
        
        total_loss = (1.0 - supcon_weight) * infonce_loss + supcon_weight * supcon_loss
        return total_loss, infonce_loss.detach(), supcon_loss.detach()