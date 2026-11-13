import os
import sys
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

from models.multimodal import BRSETMultimodalSSL
from dataset import BRSETMultimodalDataset
from torchvision import transforms


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def apply_gpu_key_augmentation(images):
    """
    Creates an independent augmented view for the Momentum Key Encoder
    directly on GPU with minimal memory overhead.
    """
    aug_imgs = images.clone()
    if torch.rand(1).item() < 0.5:
        aug_imgs = torch.flip(aug_imgs, dims=[3])
    if torch.rand(1).item() < 0.5:
        aug_imgs = torch.flip(aug_imgs, dims=[2])
    scale = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * 0.15
    shift = (torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5) * 0.10
    return aug_imgs * scale + shift


class HybridMoCoVICRegEngine(nn.Module):
    """
    NaN-Safe Hybrid MoCo-v2 + Scale-Aware VICReg Multimodal SSL Engine
    with Tabular False-Negative Collision Masking.
    """
    def __init__(self, embed_dim=256, queue_size=2048, temperature=0.10, fn_sim_threshold=0.98):
        super().__init__()
        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.fn_sim_threshold = fn_sim_threshold

        self.register_buffer("img_queue", torch.randn(embed_dim, queue_size, dtype=torch.float32))
        self.register_buffer("tab_queue", torch.randn(embed_dim, queue_size, dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.img_queue = F.normalize(self.img_queue, dim=0, eps=1e-6)
        self.tab_queue = F.normalize(self.tab_queue, dim=0, eps=1e-6)

    @torch.no_grad()
    def update_momentum_encoder(self, query_model, key_model, m=0.996):
        for param_q, param_k in zip(query_model.parameters(), key_model.parameters()):
            if torch.isfinite(param_q.data).all():
                param_k.data.mul_(m).add_(param_q.detach().data, alpha=1.0 - m)

    @torch.no_grad()
    def enqueue_dequeue(self, key_img_emb, key_tab_emb):
        if not (torch.isfinite(key_img_emb).all() and torch.isfinite(key_tab_emb).all()):
            return

        batch_size = key_img_emb.shape[0]
        ptr = int(self.queue_ptr)

        img_norm = F.normalize(key_img_emb.detach().float(), dim=1, eps=1e-6)
        tab_norm = F.normalize(key_tab_emb.detach().float(), dim=1, eps=1e-6)

        if ptr + batch_size <= self.queue_size:
            self.img_queue[:, ptr:ptr + batch_size] = img_norm.T
            self.tab_queue[:, ptr:ptr + batch_size] = tab_norm.T
        else:
            rem = self.queue_size - ptr
            self.img_queue[:, ptr:] = img_norm[:rem].T
            self.tab_queue[:, ptr:] = tab_norm[:rem].T
            self.img_queue[:, :batch_size - rem] = img_norm[rem:].T
            self.tab_queue[:, :batch_size - rem] = tab_norm[rem:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def compute_vicreg_var_cov_loss(self, z_img, z_tab):
        batch_size, num_features = z_img.shape
        target_std = 1.0 / math.sqrt(num_features)

        std_img = torch.sqrt(z_img.var(dim=0, unbiased=False) + 1e-4)
        std_tab = torch.sqrt(z_tab.var(dim=0, unbiased=False) + 1e-4)
        var_loss = 0.5 * (
            torch.mean(F.relu(target_std - std_img)) +
            torch.mean(F.relu(target_std - std_tab))
        ) / target_std

        z_img_centered = z_img - z_img.mean(dim=0, keepdim=True)
        z_tab_centered = z_tab - z_tab.mean(dim=0, keepdim=True)
        cov_img = (z_img_centered.T @ z_img_centered) / max(batch_size - 1, 1)
        cov_tab = (z_tab_centered.T @ z_tab_centered) / max(batch_size - 1, 1)

        scale_factor = float(num_features)
        cov_loss = scale_factor * (
            off_diagonal(cov_img).pow(2).sum() + off_diagonal(cov_tab).pow(2).sum()
        ) / num_features

        return var_loss + cov_loss

    def compute_hybrid_loss(self, q_img, q_tab, k_img, k_tab, vicreg_weight=0.15):
        q_img_norm = F.normalize(q_img.float(), dim=1, eps=1e-6)
        q_tab_norm = F.normalize(q_tab.float(), dim=1, eps=1e-6)
        k_img_norm = F.normalize(k_img.detach().float(), dim=1, eps=1e-6)
        k_tab_norm = F.normalize(k_tab.detach().float(), dim=1, eps=1e-6)

        vicreg_loss = self.compute_vicreg_var_cov_loss(q_img_norm, q_tab_norm)

        batch_size = q_img_norm.shape[0]
        targets = torch.arange(batch_size, dtype=torch.long, device=q_img_norm.device)

        queue_tab_detached = self.tab_queue.clone().detach()
        queue_img_detached = self.img_queue.clone().detach()

        # False-Negative Masking: mask queue entries whose clinical table is nearly identical to current patient
        tab_sim_queue = k_tab_norm @ queue_tab_detached
        fn_mask = tab_sim_queue > self.fn_sim_threshold

        # 1. Cross-Modal MoCo Contrast (Image <-> Tabular)
        logits_i2t_batch = (q_img_norm @ k_tab_norm.T) / self.temperature
        logits_t2i_batch = (q_tab_norm @ k_img_norm.T) / self.temperature

        logits_i2t_queue = (q_img_norm @ queue_tab_detached) / self.temperature
        logits_t2i_queue = (q_tab_norm @ queue_img_detached) / self.temperature

        logits_i2t_queue = logits_i2t_queue.masked_fill(fn_mask, -1e4)
        logits_t2i_queue = logits_t2i_queue.masked_fill(fn_mask, -1e4)

        logits_i2t = torch.cat([logits_i2t_batch, logits_i2t_queue], dim=1)
        logits_t2i = torch.cat([logits_t2i_batch, logits_t2i_queue], dim=1)

        loss_cross = 0.5 * (F.cross_entropy(logits_i2t, targets) + F.cross_entropy(logits_t2i, targets))

        # 2. Intra-Modal Vision MoCo Contrast (Query Image <-> Momentum Key Image)
        logits_i2i_batch = (q_img_norm @ k_img_norm.T) / self.temperature
        logits_i2i_queue = (q_img_norm @ queue_img_detached) / self.temperature
        logits_i2i = torch.cat([logits_i2i_batch, logits_i2i_queue], dim=1)
        loss_intra_img = F.cross_entropy(logits_i2i, targets)

        moco_loss = 0.6 * loss_cross + 0.4 * loss_intra_img
        total_loss = moco_loss + vicreg_weight * vicreg_loss

        return total_loss, moco_loss, vicreg_loss


def load_config(config_path="config_dr3_v2.yaml"):
    with open(os.path.join(CURRENT_DIR, config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_path(path_str):
    path_str = str(path_str)
    if os.path.isabs(path_str):
        return path_str
    while "../" in path_str:
        path_str = path_str.replace("../", "")
    return path_str.strip("/")


def save_checkpoint(state, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)


def main():
    torch.backends.cudnn.benchmark = True
    config = load_config("config_dr3_v2.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_img_size = int(config['dataset'].get('ssl_image_size', 448))
    batch_size = int(config['ssl_training']['batch_size'])
    grad_accum_steps = int(config['ssl_training'].get('grad_accum_steps', 16))
    queue_size = int(config['ssl_training'].get('queue_size', 2048))
    moco_m = float(config['ssl_training'].get('moco_momentum', 0.996))
    vicreg_w = float(config['ssl_training'].get('vicreg_weight', 0.15))
    fn_sim_thresh = float(config['ssl_training'].get('fn_sim_threshold', 0.98))
    warmup_epochs = int(config['ssl_training'].get('warmup_queue_epochs', 2))
    modality_dropout_p = float(config['fusion'].get('modality_dropout_p', 0.25))
    fixed_temp = float(config['model'].get('temperature', 0.10))

    print(f"🚀 Training Hybrid MoCo-v2 + VICReg Multimodal SSL (FN-Masking) | Size: {ssl_img_size}x{ssl_img_size}")
    print(f"🔒 Label-Free SSL | Batch: {batch_size} (Accum: {grad_accum_steps}) | Queue: {queue_size} | EMA: {moco_m}")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))
    vision_weights = os.path.join(PROJECT_ROOT, clean_path(config['model']['vision_weights']))
    checkpoint_dir = os.path.join(PROJECT_ROOT, clean_path(config['ssl_training']['checkpoint_dir']))
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((ssl_img_size, ssl_img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transform=transform,
        task_type="ssl",
        is_train=True,
        modality_dropout_p=modality_dropout_p
    )

    num_workers = int(config['ssl_training'].get('num_workers', 4))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )

    embed_dim = int(config['model']['embed_dim'])
    num_tab = int(config['dataset']['num_tabular_features'])

    # 1. Online Query Encoder
    model = BRSETMultimodalSSL(
        embed_dim=embed_dim,
        num_tabular_features=num_tab,
        temperature=fixed_temp,
        vision_weights_path=vision_weights
    ).to(device)

    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = False

    # 2. Momentum Key Encoder (Clean instantiation without deepcopy hook conflicts)
    key_model = BRSETMultimodalSSL(
        embed_dim=embed_dim,
        num_tabular_features=num_tab,
        temperature=fixed_temp,
        vision_weights_path=vision_weights
    ).to(device)
    key_model.load_state_dict(model.state_dict())
    for param in key_model.parameters():
        param.requires_grad = False
    key_model.train()

    # Clear unused forward hooks during SSL pretraining to save 4GB VRAM
    for m in [model, key_model]:
        for submodule in m.vision_encoder.modules():
            if hasattr(submodule, '_forward_hooks'):
                submodule._forward_hooks.clear()

    ssl_engine = HybridMoCoVICRegEngine(
        embed_dim=embed_dim,
        queue_size=queue_size,
        temperature=fixed_temp,
        fn_sim_threshold=fn_sim_thresh
    ).to(device)

    num_epochs = int(config['ssl_training']['epochs'])
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config['ssl_training']['learning_rate']),
        weight_decay=float(config['ssl_training']['weight_decay'])
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    history_file = os.path.join(checkpoint_dir, "ssl_history_moco_vicreg.csv")
    start_epoch = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if config['ssl_training'].get('resume_from_checkpoint', False) and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if math.isfinite(checkpoint.get('best_loss', float('inf'))):
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'key_model_state_dict' in checkpoint:
                key_model.load_state_dict(checkpoint['key_model_state_dict'])
            if 'ssl_engine_state_dict' in checkpoint:
                ssl_engine.load_state_dict(checkpoint['ssl_engine_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler.T_max = num_epochs
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"🔄 Resumed Hybrid MoCo-VICReg from Epoch {start_epoch} (Best Loss: {best_loss:.4f} | No Improve: {epochs_no_improve})")
        else:
            print("⚠️ Found corrupted NaN checkpoint; starting fresh from Epoch 1.")
            with open(history_file, "w", encoding="utf-8") as f:
                f.write("epoch,lr,total_loss,moco_loss,vicreg_loss,is_best,no_improve_count\n")
    else:
        with open(history_file, "w", encoding="utf-8") as f:
            f.write("epoch,lr,total_loss,moco_loss,vicreg_loss,is_best,no_improve_count\n")

    patience = int(config['ssl_training']['early_stopping_patience'])

    for epoch in range(start_epoch, num_epochs):
        model.train()
        key_model.train()
        total_loss_sum, moco_sum, vicreg_sum = 0.0, 0.0, 0.0
        valid_steps = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(dataloader, desc=f"MoCo-VICReg Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, batch in enumerate(progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                q_img, q_tab = model(images, tabular, masks)
                with torch.no_grad():
                    key_images = apply_gpu_key_augmentation(images)
                    clean_masks = torch.zeros_like(masks)
                    k_img, k_tab = key_model(key_images, tabular, clean_masks)

            loss, moco_l, vicreg_l = ssl_engine.compute_hybrid_loss(
                q_img, q_tab, k_img, k_tab, vicreg_weight=vicreg_w
            )

            if torch.isfinite(loss):
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                ssl_engine.enqueue_dequeue(k_img, k_tab)

                total_loss_sum += loss.item()
                moco_sum += moco_l.item()
                vicreg_sum += vicreg_l.item()
                valid_steps += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ssl_engine.update_momentum_encoder(model, key_model, m=moco_m)

            if valid_steps > 0:
                progress_bar.set_postfix({
                    'loss': f"{total_loss_sum / valid_steps:.4f}",
                    'moco': f"{moco_sum / valid_steps:.4f}",
                    'vic': f"{vicreg_sum / valid_steps:.4f}"
                })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        denom = max(valid_steps, 1)
        avg_loss = total_loss_sum / denom
        avg_moco = moco_sum / denom
        avg_vicreg = vicreg_sum / denom

        if (epoch + 1) <= warmup_epochs:
            is_best = False
            epochs_no_improve = 0
            status_msg = f" ⏳ [Queue Warm-up {epoch+1}/{warmup_epochs}]"
            if (epoch + 1) == warmup_epochs:
                best_loss = avg_loss
                is_best = True
                status_msg = " 🌟 [Baseline Best Set]"
        else:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                epochs_no_improve = 0
                status_msg = " 🌟 [Best Saved]"
            else:
                epochs_no_improve += 1
                status_msg = f" ⚠️ [No Improvement: {epochs_no_improve}/{patience}] (Best: {best_loss:.4f})"

        with open(history_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{current_lr:.6e},{avg_loss:.4f},{avg_moco:.4f},"
                f"{avg_vicreg:.4f},{int(is_best)},{epochs_no_improve}\n"
            )

        print(
            f"🏁 Epoch {epoch+1:03d}/{num_epochs} | Total: {avg_loss:.4f} "
            f"(MoCo-v2: {avg_moco:.4f}, VICReg: {avg_vicreg:.4f}){status_msg}"
        )

        if (epoch + 1) % config['ssl_training']['save_every_n_epochs'] == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'key_model_state_dict': key_model.state_dict(),
                'ssl_engine_state_dict': ssl_engine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'epochs_no_improve': epochs_no_improve,
                'config': config
            }
            save_checkpoint(checkpoint_state, checkpoint_dir, is_best)

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break


if __name__ == "__main__":
    main()