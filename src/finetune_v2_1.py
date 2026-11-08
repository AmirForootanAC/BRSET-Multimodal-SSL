import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from dataset_v2 import BRSETMultimodalDatasetV2
from models.multimodal_v2 import BRSETMultimodalSSL_V2
from torchvision import transforms
from utils.metrics import calculate_metrics

def load_config(config_path="../config_v2.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_pos_weights(dataset, train_indices):
    train_labels = dataset.df.iloc[train_indices][dataset.target_cols].values.astype(float)
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)
    soft_weights = np.power(raw_weights, 0.3)
    return torch.tensor(soft_weights, dtype=torch.float32)

class BRSETClassifier_V2(nn.Module):
    def __init__(self, embed_dim=256, num_tabular_features=6, num_classes=10):
        super(BRSETClassifier_V2, self).__init__()
        
        self.ssl_base = BRSETMultimodalSSL_V2(
            embed_dim=embed_dim, 
            num_tabular_features=num_tabular_features
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def load_ssl_weights(self, checkpoint_path, device):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            self.ssl_base.load_state_dict(state_dict)
            print(f"[SUCCESS] Pre-trained SSL V2 weights fully loaded from {checkpoint_path}")
        else:
            print(f"[WARNING] SSL checkpoint not found at {checkpoint_path}")

    def forward(self, image, tabular, mask):
        img_emb, tab_emb = self.ssl_base(image, tabular, mask)
        fused_features = torch.cat((img_emb, tab_emb), dim=1)
        logits = self.classifier(fused_features)
        return logits

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multi-Task Fine-tuning (V2) on {device} with AMP (Optimized)")

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDatasetV2(
        csv_file=f"../{config['dataset']['csv_path']}",
        img_dir=f"../{config['dataset']['img_dir']}",
        transform=transform
    )
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    batch_size = config['finetuning']['batch_size']
    num_workers = config['training']['num_workers']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pos_weights = compute_pos_weights(dataset, train_dataset.indices).to(device)

    model = BRSETClassifier_V2(
        embed_dim=config['model']['embed_dim'],
        num_tabular_features=config['dataset']['num_tabular_features'],
        num_classes=10
    ).to(device)
    
    ssl_checkpoint_path = os.path.join(f"../{config['training']['checkpoint_dir']}", "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    ssl_params = model.ssl_base.parameters()
    head_params = model.classifier.parameters()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # FIXED: Explicitly set Learning Rates
    optimizer = AdamW([
        {'params': ssl_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=config['finetuning']['weight_decay'])
    
    num_epochs = config['finetuning']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(device.type)

    log_dir = f"../{config['finetuning']['save_dir']}"
    os.makedirs(log_dir, exist_ok=True)
    
    # FIXED: Track Best AUROC instead of Best Loss
    best_val_auroc = 0.0
    patience = 20 # INCREASED patience
    epochs_no_improve = 0
    disease_names = ["Diabetic Retinopathy", "Glaucoma", "Macular Edema", "Pathological Myopia", "AMD"]
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.empty_cache()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                tabular = batch['tabular'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device).float()
                
                with autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images, tabular, masks)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu().float())
                all_preds.append(probs.cpu().float())
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0)
        y_pred = torch.cat(all_preds, dim=0)
        
        y_true_main = y_true[:, :5]
        y_pred_main = y_pred[:, :5]
        
        metrics = calculate_metrics(y_true_main, y_pred_main, task_type="multilabel")
        b_accs = metrics['per_class_balanced_acc']
        current_auroc = metrics['auroc']

        # FIXED: Save model based on highest AUROC
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            save_path = os.path.join(log_dir, "best_multilabel_model_v2.pth")
            torch.save(model.state_dict(), save_path)
            best_model_msg = f" | 🌟 NEW BEST MODEL (AUROC: {best_val_auroc:.4f})"
            epochs_no_improve = 0
        else:
            best_model_msg = ""
            epochs_no_improve += 1

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_model_msg}")
        print(f"Overall AUROC: {current_auroc:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
        print("-" * 50)
        for name, b_acc in zip(disease_names, b_accs):
            print(f"  - {name}: {b_acc:.4f}")
        print("-" * 50)

        if epochs_no_improve >= patience:
            print(f"Early stopping! No improvement in AUROC for {patience} epochs.")
            break

    print("\n" + "="*60)
    print("Evaluating BEST model on TEST set...")
    best_model_path = os.path.join(log_dir, "best_multilabel_model_v2.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_labels_test, all_preds_test = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Set"):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device).float()
            
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images, tabular, masks)
            
            probs = torch.sigmoid(logits)
            all_labels_test.append(labels.cpu().float())
            all_preds_test.append(probs.cpu().float())
            
    y_true_test = torch.cat(all_labels_test, dim=0)[:, :5]
    y_pred_test = torch.cat(all_preds_test, dim=0)[:, :5]
    
    test_metrics = calculate_metrics(y_true_test, y_pred_test, task_type="multilabel")
    test_b_accs = test_metrics['per_class_balanced_acc']
    
    print(f"\nFINAL TEST RESULTS V2")
    print(f"Overall AUROC: {test_metrics['auroc']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("-" * 50)
    for name, b_acc in zip(disease_names, test_b_accs):
        print(f"  - {name}: {b_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()