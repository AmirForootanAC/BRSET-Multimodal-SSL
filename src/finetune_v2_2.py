import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler

from dataset_v2 import BRSETMultimodalDatasetV2
from models.multimodal_v2 import BRSETMultimodalSSL_V2
from torchvision import transforms
from utils.metrics import calculate_metrics

def load_config(config_path="../config_v2.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_pos_weights(dataset, train_indices):
    target_cols = ['diabetic_retinopathy', 'increased_cup_disc', 'macular_edema', 'myopic_fundus', 'amd']
    if all(col in dataset.df.columns for col in target_cols):
        train_labels = dataset.df.iloc[train_indices][target_cols].values.astype(float)
    else:
        train_labels = dataset.df.iloc[train_indices][dataset.target_cols[:5]].values.astype(float)
        
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)
    soft_weights = np.power(raw_weights, 0.3)
    return torch.tensor(soft_weights, dtype=torch.float32)

def filter_adequate_quality(dataset):
    """
    Replicates the Image Quality Check from Raghu et al. (arXiv:2507.12669)
    by filtering out low-quality/Inadequate fundus images.
    """
    df = dataset.df
    initial_len = len(df)
    
    # Case 1: Check direct quality column (Standard BRSET)
    for col in ['quality', 'Quality', 'image_quality']:
        if col in df.columns:
            # Keep images with Adequate or 1
            vals = df[col].astype(str).str.strip().str.lower()
            mask = vals.isin(['adequate', '1', '1.0', 'good', 'true', 'normal'])
            if mask.sum() > 1000:
                dataset.df = df[mask].reset_index(drop=True)
                print(f"[Quality Filter] Active on '{col}': {initial_len} -> {len(dataset.df)} clean images kept ({initial_len - len(dataset.df)} low-quality removed).")
                return

    # Case 2: If quality column is missing, check the 4 detailed QC columns
    qc_cols = ['focus', 'illumination', 'image_field', 'artifacts']
    if all(c in df.columns for c in qc_cols):
        # In BRSET, 1 for focus/illumination/image_field means adequate.
        # We use mode to find the majority "clean" label format.
        mask = (df['focus'] == df['focus'].mode()[0]) & \
               (df['illumination'] == df['illumination'].mode()[0]) & \
               (df['image_field'] == df['image_field'].mode()[0]) & \
               (df['artifacts'] == df['artifacts'].mode()[0])
        if mask.sum() > 1000:
            dataset.df = df[mask].reset_index(drop=True)
            print(f"[Quality Filter] Active on 4 QC columns: {initial_len} -> {len(dataset.df)} clean images kept ({initial_len - len(dataset.df)} removed).")
            return
            
    print(f"[Quality Filter WARNING] Could not find quality columns in CSV! Columns available: {list(df.columns)}")

class RawBackboneExtractor(nn.Module):
    """
    Hooks into DualVisionEncoder to grab the raw 3072-dim (ResNet50 + DenseNet121)
    pooled features BEFORE the collapsed 256-dim SSL projection head.
    """
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.raw_features = {}
        self.hook_registered = False
        
        for name, module in self.vision_encoder.named_modules():
            if isinstance(module, nn.Linear) and module.in_features >= 1024:
                module.register_forward_hook(self._save_input_hook)
                self.in_features = module.in_features
                self.hook_registered = True
                print(f"[Bypass Hook] Attached to '{name}' | Extracted Raw Vision Dim: {self.in_features} (Bypassing 256-d bottleneck)")
                break
                
        if not self.hook_registered:
            self.in_features = 256
            print("[Bypass Hook] Could not find wide Linear layer; using default 256-d output.")

    def _save_input_hook(self, module, input, output):
        self.raw_features['feat'] = input[0]

    def forward(self, x):
        out = self.vision_encoder(x)
        if self.hook_registered and 'feat' in self.raw_features:
            return self.raw_features['feat']
        return out

class BRSETClassifier_V2_2(nn.Module):
    def __init__(self, embed_dim=256, num_tabular_features=6, num_classes=5):
        super(BRSETClassifier_V2_2, self).__init__()
        
        self.ssl_base = BRSETMultimodalSSL_V2(
            embed_dim=embed_dim, 
            num_tabular_features=num_tabular_features
        )
        
        self.vision_extractor = RawBackboneExtractor(self.ssl_base.vision_encoder)
        vis_dim = self.vision_extractor.in_features
        
        # Projection & Fusion Head with LayerNorm to balance Vision + Tabular scales
        self.vis_norm = nn.LayerNorm(vis_dim)
        self.tab_norm = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(vis_dim + embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def load_ssl_weights(self, checkpoint_path, device):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.ssl_base.load_state_dict(state_dict, strict=False)
            print(f"[SUCCESS] Pre-trained SSL V2 weights loaded from {checkpoint_path}")
        else:
            print(f"[WARNING] SSL checkpoint not found at {checkpoint_path}. Using base FLAIR + DenseNet weights!")

    def forward(self, image, tabular, mask):
        vis_raw = self.vision_extractor(image)
        tab_emb = self.ssl_base.tabular_encoder(tabular, mask)
        
        vis_normed = self.vis_norm(vis_raw)
        tab_normed = self.tab_norm(tab_emb)
        
        fused = torch.cat((vis_normed, tab_normed), dim=1)
        logits = self.classifier(fused)
        return logits

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting V2.2 Fast 10-Epoch Benchmark (5 Diseases | FP32 | Raw 3072-D Bypass) on {device}")

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
    
    # Optional Quality Filtering (Set to True to match Raghu et al.'s clean-image evaluation)
    APPLY_QUALITY_FILTER = True
    if APPLY_QUALITY_FILTER:
        filter_adequate_quality(dataset)
    
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

    print(f"Dataset split: {train_size} Train | {val_size} Val | {test_size} Test")

    pos_weights = compute_pos_weights(dataset, train_dataset.indices).to(device)
    print(f"Pos Weights (5 Diseases): {[round(x, 3) for x in pos_weights.cpu().tolist()]}")

    model = BRSETClassifier_V2_2(
        embed_dim=config['model']['embed_dim'],
        num_tabular_features=config['dataset']['num_tabular_features'],
        num_classes=5
    ).to(device)
    
    ssl_checkpoint_path = os.path.join(f"../{config['training']['checkpoint_dir']}", "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    # Separate parameters: 1e-6 for SSL backbones (like V1), 1e-3 for new Classifier & Norms
    ssl_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ssl_base' in name:
            ssl_params.append(param)
        else:
            head_params.append(param)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    head_lr = config['finetuning'].get('learning_rate', 1e-3)
    optimizer = AdamW([
        {'params': ssl_params, 'lr': 1e-6},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=config['finetuning']['weight_decay'])

    log_dir = f"../{config['finetuning']['save_dir']}"
    os.makedirs(log_dir, exist_ok=True)
    history_csv = os.path.join(log_dir, "multilabel_v2_2_history.csv")
    
    with open(history_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,auroc,macro_f1,dr_bacc,glaucoma_bacc,dme_bacc,myopia_bacc,amd_bacc\n")

    best_val_auroc = 0.0
    num_epochs = 10  # 10-Epoch Benchmark Test
    disease_names = ["Diabetic Retinopathy", "Glaucoma (Cup/Disc)", "Macular Edema", "Pathological Myopia", "AMD"]

    scaler = GradScaler(device.type)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            # Crucial Fix: Train strictly on the 5 main diseases
            labels = batch['labels'][:, :5].to(device).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use AMP to save VRAM and speed up training
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Clear cache to prevent memory accumulation
            torch.cuda.empty_cache()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
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
                labels = batch['labels'][:, :5].to(device).float()
                
                # Use AMP during validation to reduce VRAM usage
                with autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images, tabular, masks)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu())
                all_preds.append(probs.cpu())
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0).float()
        y_pred = torch.cat(all_preds, dim=0).float()
        
        metrics = calculate_metrics(y_true, y_pred, task_type="multilabel")
        b_accs = metrics['per_class_balanced_acc']
        current_auroc = metrics['auroc']
        
        with open(history_csv, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{current_auroc:.4f},{metrics['macro_f1']:.4f},"
                    f"{b_accs[0]:.4f},{b_accs[1]:.4f},{b_accs[2]:.4f},{b_accs[3]:.4f},{b_accs[4]:.4f}\n")

        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            save_path = os.path.join(log_dir, "best_multilabel_model_v2_2.pth")
            torch.save(model.state_dict(), save_path)
            best_model_msg = f" | 🌟 Best Model Saved (AUROC: {best_val_auroc:.4f})"
        else:
            best_model_msg = ""

        print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_model_msg}")
        print(f"Overall AUROC: {current_auroc:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
        
        opt_thresh = metrics.get('optimal_thresholds', [0.5]*5)
        print(f"Optimal Thresholds applied: {[round(float(t), 3) for t in opt_thresh]}")
        
        print("-" * 50)
        print("Val Set Balanced Accuracy Per Disease:")
        for name, b_acc in zip(disease_names, b_accs):
            print(f"  - {name}: {b_acc:.4f}")
        print("-" * 50)

    print("\n" + "="*60)
    print("10-Epoch Benchmark Complete. Evaluating BEST model on TEST set...")
    print("="*60)
    
    best_model_path = os.path.join(log_dir, "best_multilabel_model_v2_2.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    model.eval()
    test_loss = 0.0
    all_labels_test, all_preds_test = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'][:, :5].to(device).float()
            
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
                
            test_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_labels_test.append(labels.cpu())
            all_preds_test.append(probs.cpu())
            
    avg_test_loss = test_loss / len(test_loader)
    y_true_test = torch.cat(all_labels_test, dim=0).float()
    y_pred_test = torch.cat(all_preds_test, dim=0).float()
    
    test_metrics = calculate_metrics(y_true_test, y_pred_test, task_type="multilabel")
    test_b_accs = test_metrics['per_class_balanced_acc']
    
    print(f"\nFINAL 10-EPOCH TEST RESULTS V2.2 | Test Loss: {avg_test_loss:.4f}")
    print(f"Overall AUROC: {test_metrics['auroc']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    opt_thresh_test = test_metrics.get('optimal_thresholds', [0.5]*5)
    print(f"Optimal Thresholds applied (Test): {[round(float(t), 3) for t in opt_thresh_test]}")
    print("-" * 50)
    print("Test Set Balanced Accuracy Per Disease (Compare with MetaFusion):")
    for name, b_acc in zip(disease_names, test_b_accs):
        print(f"  - {name}: {b_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()