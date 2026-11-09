import os
import sys
import yaml
import torch
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

def load_config(config_path="config_dr3.yaml"):
    with open(os.path.join(CURRENT_DIR, config_path), "r") as f:
        return yaml.safe_load(f)

def clean_path(path_str):
    path_str = str(path_str)
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
    config = load_config("config_dr3.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_img_size = int(config['dataset'].get('ssl_image_size', 224))
    batch_size = int(config['ssl_training']['batch_size'])
    grad_accum_steps = int(config['ssl_training'].get('grad_accum_steps', 2))
    supcon_weight = float(config['ssl_training'].get('supcon_weight', 0.4))
    modality_dropout_p = float(config['fusion'].get('modality_dropout_p', 0.25))

    print(f"🚀 Training Multimodal SSL (Image Size: {ssl_img_size}x{ssl_img_size}) | Device: {device}")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))
    vision_weights = os.path.join(PROJECT_ROOT, clean_path(config['model']['vision_weights']))
    checkpoint_dir = os.path.join(PROJECT_ROOT, clean_path(config['ssl_training']['checkpoint_dir']))
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((ssl_img_size, ssl_img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
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

    model = BRSETMultimodalSSL(
        embed_dim=int(config['model']['embed_dim']),
        num_tabular_features=int(config['dataset']['num_tabular_features']),
        temperature=float(config['model']['temperature']),
        vision_weights_path=vision_weights
    ).to(device)

    num_epochs = int(config['ssl_training']['epochs'])
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config['ssl_training']['learning_rate']), 
        weight_decay=float(config['ssl_training']['weight_decay'])
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    history_file = os.path.join(checkpoint_dir, "ssl_history.csv")
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    if config['ssl_training'].get('resume_from_checkpoint', True) and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"🔄 Resumed SSL training from Epoch {start_epoch} (Best Loss: {best_loss:.4f})")
    else:
        with open(history_file, "w") as f:
            f.write("epoch,lr,total_loss,infonce_loss,supcon_loss,temperature,is_best\n")

    patience = int(config['ssl_training']['early_stopping_patience'])
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss_sum, infonce_sum, supcon_sum = 0.0, 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)
        
        progress_bar = tqdm(dataloader, desc=f"SSL Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, batch in enumerate(progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                img_emb, tab_emb = model(images, tabular, masks)
                loss, infonce_l, supcon_l = model.compute_contrastive_loss(
                    img_emb, tab_emb, labels=labels, supcon_weight=supcon_weight
                )
                scaled_loss = loss / grad_accum_steps
            
            scaler.scale(scaled_loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss_sum += loss.item()
            infonce_sum += infonce_l.item()
            supcon_sum += supcon_l.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        avg_loss = total_loss_sum / len(dataloader)
        avg_infonce = infonce_sum / len(dataloader)
        avg_supcon = supcon_sum / len(dataloader)
        current_temp = (1.0 / model.logit_scale.exp().clamp(max=100.0)).item()
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_msg = " 🌟 [Best Saved]"
        else:
            epochs_no_improve += 1
            best_msg = ""
            
        with open(history_file, "a") as f:
            f.write(
                f"{epoch+1},{current_lr:.6e},{avg_loss:.4f},{avg_infonce:.4f},"
                f"{avg_supcon:.4f},{current_temp:.5f},{int(is_best)}\n"
            )
            
        print(
            f"🏁 Epoch {epoch+1:03d}/{num_epochs} | Total Loss: {avg_loss:.4f} "
            f"(InfoNCE: {avg_infonce:.4f}, SupCon: {avg_supcon:.4f}) | Temp: {current_temp:.4f}{best_msg}"
        )
        
        if (epoch + 1) % config['ssl_training']['save_every_n_epochs'] == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }
            save_checkpoint(checkpoint_state, checkpoint_dir, is_best)

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break

if __name__ == "__main__":
    main()