import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.multimodal_v2 import BRSETMultimodalSSL_V2
from dataset_v2 import BRSETMultimodalDatasetV2
from torchvision import transforms

def load_config(config_path="../config_v2.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(state, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def main():
    config = load_config("../config_v2.yaml")
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Training Multimodal SSL V2 on device: {device} (OOM-Safe + WarmUp)")

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDatasetV2(
        csv_file=f"../{config['dataset']['csv_path']}",
        img_dir=f"../{config['dataset']['img_dir']}",
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers'],
        pin_memory=True 
    )

    model = BRSETMultimodalSSL_V2(
        embed_dim=config['model']['embed_dim'],
        num_tabular_features=config['dataset']['num_tabular_features'],
        temperature=config['model'].get('temperature', 0.1)
    ).to(device)

    # Switched to Standard Adam for better initial contrastive stabilization
    optimizer = Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    scaler = GradScaler(device_type)
    
    num_epochs = config['training']['epochs']
    total_steps = len(dataloader) * num_epochs
    warmup_steps = len(dataloader) * 5 # 5 epochs of warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = f"../{config['training']['checkpoint_dir']}"
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    if config['training'].get('resume_from_checkpoint', False) and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resumed training from Epoch {start_epoch} | Current Best Loss: {best_loss:.4f}")
    else:
        print("Starting training from scratch.")

    # Increased patience due to heavy ensembling
    patience = max(config['training'].get('early_stopping_patience', 10), 15) 
    save_every_n = config['training'].get('save_every_n_epochs', 5)
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [SSL]", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device_type, dtype=torch.float16):
                img_emb, tab_emb = model(images, tabular, masks)
                loss = model.compute_contrastive_loss(img_emb, tab_emb)
            
            scaler.scale(loss).backward()
            
            # ثبت مقیاس فعلیِ scaler قبل از آپدیت
            scale_before = scaler.get_scale()
            
            scaler.step(optimizer)
            scaler.update()
            
            # اگر scaler به خاطر مقادیر بی‌نهایت (NaN/Inf) آپدیت را رد نکرده بود، scheduler یک قدم جلو می‌رود
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
            
            # جلوگیری از خطای پر شدن حافظه گرافیک (OOM)
            torch.cuda.empty_cache()
            
            total_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % save_every_n == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }
            save_checkpoint(checkpoint_state, checkpoint_dir, is_best)
            
            if is_best:
                print(f"New best SSL model saved! (Loss: {best_loss:.4f})")
            else:
                print(f"Periodic checkpoint saved at Epoch {epoch+1}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break

if __name__ == "__main__":
    main()