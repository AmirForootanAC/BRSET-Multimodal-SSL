import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append('..') 

from models.multimodal import BRSETMultimodalSSL
from dataset import BRSETMultimodalDataset
from torchvision import transforms

def load_config(config_path="config_dr3.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(state, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)

def main():
    config = load_config("config_dr3.yaml")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Multimodal SSL (8 Features) | Run ID: {run_id} | Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDataset(
        csv_file=config['dataset']['csv_path'],
        img_dir=config['dataset']['img_dir'],
        transform=transform,
        task_type="ssl"
    )
    
    dataloader = DataLoader(
        dataset, batch_size=config['ssl_training']['batch_size'], 
        shuffle=True, num_workers=config['ssl_training']['num_workers'], pin_memory=False 
    )

    model = BRSETMultimodalSSL(
        embed_dim=config['model']['embed_dim'],
        num_tabular_features=config['dataset']['num_tabular_features'],
        temperature=config['model']['temperature']
    ).to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr=config['ssl_training']['learning_rate'], 
        weight_decay=config['ssl_training']['weight_decay']
    )
    scaler = GradScaler(device.type)

    # History Log Initialization
    log_dir = config['ssl_training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    history_file = os.path.join(log_dir, f"ssl_history_{run_id}.csv")
    with open(history_file, "w") as f:
        f.write("epoch,loss\n")

    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(config['ssl_training']['checkpoint_dir'], "latest_checkpoint.pth")
    
    if config['ssl_training']['resume_from_checkpoint'] and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"🔄 Resumed training from Epoch {start_epoch}")

    num_epochs = config['ssl_training']['epochs']
    patience = config['ssl_training']['early_stopping_patience']
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=False)
            tabular = batch['tabular'].to(device, non_blocking=False)
            masks = batch['mask'].to(device, non_blocking=False)
            
            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=torch.float16):
                img_emb, tab_emb = model(images, tabular, masks)
                loss = model.compute_contrastive_loss(img_emb, tab_emb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        
        # Save to history file
        with open(history_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f}\n")
            
        print(f"🏁 Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % config['ssl_training']['save_every_n_epochs'] == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }
            save_checkpoint(checkpoint_state, config['ssl_training']['checkpoint_dir'], is_best)
            if is_best:
                print(f"🌟 New best SSL model saved! (Loss: {best_loss:.4f})")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break

if __name__ == "__main__":
    main()