import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset import BRSETMultimodalDataset
from models.fusion import BRSETClassifier
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_config(config_path="config_dr3.yaml"):
    with open(os.path.join(CURRENT_DIR, config_path), "r") as f:
        return yaml.safe_load(f)

def clean_path(path_str):
    path_str = str(path_str)
    while "../" in path_str:
        path_str = path_str.replace("../", "")
    return path_str.strip("/")

def main():
    torch.backends.cudnn.benchmark = True
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    label_percent = float(config['finetuning'].get('label_percent', 100.0))
    label_fraction = label_percent / 100.0
    img_size = int(config['dataset'].get('finetune_image_size', 448))
    batch_size = int(config['finetuning']['batch_size'])
    grad_accum_steps = int(config['finetuning'].get('grad_accum_steps', 2))
    num_workers = int(config['finetuning'].get('num_workers', 4))
    modality_dropout_p = float(config['fusion'].get('modality_dropout_p', 0.25))
    
    print(f"🚀 Starting 3-Class Multimodal SSL Fine-tuning (Image Size: {img_size}x{img_size}) on {device}")
    print(f"📊 Label Efficiency Mode: Using {label_percent}% of training data")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))
    vision_weights = os.path.join(PROJECT_ROOT, clean_path(config['model']['vision_weights']))

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_ds = BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False)
    all_labels = base_ds.labels
    indices = np.arange(len(base_ds))

    # تقسیم‌بندی Stratified دقیقاً منطبق با Baseline
    train_idx, temp_idx, _, temp_labels = train_test_split(indices, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)

    if label_fraction < 1.0:
        active_train_idx, _ = train_test_split(
            train_idx,
            train_size=label_fraction,
            random_state=42,
            stratify=all_labels[train_idx]
        )
    else:
        active_train_idx = train_idx

    train_dataset = Subset(
        BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=train_transform, task_type="dr_3class", is_train=True, modality_dropout_p=modality_dropout_p),
        active_train_idx
    )
    val_dataset = Subset(
        BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False, modality_dropout_p=0.0),
        val_idx
    )
    test_dataset = Subset(
        BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False, modality_dropout_p=0.0),
        test_idx
    )

    class_counts = np.bincount(all_labels[active_train_idx])
    class_weights = torch.tensor(len(active_train_idx) / (3.0 * np.maximum(class_counts, 1)), dtype=torch.float32).to(device)

    print(f"📊 Dataset split: {len(active_train_idx)} Train | {len(val_idx)} Val | {len(test_idx)} Test")
    print(f"⚖️ Class Counts (Normal, NPDR, PDR): {class_counts.tolist()} | Weights: {[round(w, 3) for w in class_weights.tolist()]}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    model = BRSETClassifier(config, task_name="dr_3class", vision_weights_path=vision_weights).to(device)
    
    ssl_checkpoint_path = os.path.join(PROJECT_ROOT, clean_path(config['ssl_training']['checkpoint_dir']), "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    ssl_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ssl_base' in name:
            ssl_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = float(config['finetuning'].get('backbone_lr', 5e-5))
    head_lr = float(config['finetuning'].get('learning_rate', 1.5e-4))
    num_epochs = int(config['finetuning'].get('epochs', 40))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW([
        {'params': ssl_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=float(config['finetuning']['weight_decay']))
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    use_amp = bool(config['finetuning'].get('mixed_precision', True))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    base_save_dir = os.path.join(PROJECT_ROOT, clean_path(config['finetuning']['save_dir']))
    percent_str = str(int(label_percent)) if label_percent.is_integer() else str(label_percent)
    log_dir = os.path.join(base_save_dir, f"dr3_percent_{percent_str}")
    os.makedirs(log_dir, exist_ok=True)
    
    history_csv = os.path.join(log_dir, f"finetune_history_percent_{percent_str}.csv")
    save_path = os.path.join(log_dir, f"best_model_percent_{percent_str}.pth")
    test_summary_csv = os.path.join(log_dir, f"finetune_test_summary_percent_{percent_str}.csv")
    test_report_txt = os.path.join(log_dir, f"finetune_test_report_percent_{percent_str}.txt")
    
    with open(history_csv, "w") as f:
        f.write("epoch,lr_backbone,lr_head,train_loss,val_loss,val_acc,val_macro_f1,val_auroc,val_f1_normal,val_f1_npdr,val_f1_pdr,is_best\n")
        
    best_val_macro_f1 = 0.0
    patience = int(config['finetuning'].get('early_stopping_patience', 10))
    epochs_no_improve = 0
    class_names = ["Normal", "NPDR", "PDR"]
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for step, batch in enumerate(train_progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).long()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels) / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            batch_loss = loss.item() * grad_accum_steps
            train_loss += batch_loss
            train_progress_bar.set_postfix({'loss': f"{batch_loss:.4f}"})
            
        cur_lr_bb = optimizer.param_groups[0]['lr']
        cur_lr_hd = optimizer.param_groups[1]['lr']
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val  ]", leave=False)
        
        with torch.no_grad():
            for batch in val_progress_bar:
                images = batch['image'].to(device, non_blocking=True)
                tabular = batch['tabular'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True).long()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(images, tabular, masks)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                probs = torch.softmax(logits.float(), dim=1).cpu()
                all_labels.append(labels.cpu())
                all_probs.append(probs)
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()
        
        row_sums = y_prob.sum(axis=1, keepdims=True)
        y_prob = y_prob / np.maximum(row_sums, 1e-12)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        per_class_val = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        is_best = macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            best_model_msg = f" 🌟 [Best Saved: {best_val_macro_f1:.4f}]"
            epochs_no_improve = 0
        else:
            best_model_msg = ""
            epochs_no_improve += 1

        with open(history_csv, "a") as f:
            f.write(
                f"{epoch+1},{cur_lr_bb:.6e},{cur_lr_hd:.6e},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                f"{acc:.4f},{macro_f1:.4f},{auroc:.4f},"
                f"{per_class_val[0]:.4f},{per_class_val[1]:.4f},{per_class_val[2]:.4f},{int(is_best)}\n"
            )

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {macro_f1:.4f} (N:{per_class_val[0]:.2f}, NP:{per_class_val[1]:.2f}, P:{per_class_val[2]:.2f}) | "
            f"Acc: {acc:.4f} | AUROC: {auroc:.4f}{best_model_msg}"
        )

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! No improvement in Macro-F1 for {patience} consecutive epochs.")
            break

    print("\n" + "="*60)
    print("Training Complete. Evaluating BEST Multimodal Model on TEST set...")
    print("="*60)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model from {save_path}")

    model.eval()
    test_loss = 0.0
    all_labels_test, all_probs_test = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).long()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
                
            test_loss += loss.item()
            probs = torch.softmax(logits.float(), dim=1).cpu()
            all_labels_test.append(labels.cpu())
            all_probs_test.append(probs)
            
    avg_test_loss = test_loss / len(test_loader)
    y_true_test = torch.cat(all_labels_test, dim=0).numpy()
    y_prob_test = torch.cat(all_probs_test, dim=0).numpy()
    
    row_sums_t = y_prob_test.sum(axis=1, keepdims=True)
    y_prob_test = y_prob_test / np.maximum(row_sums_t, 1e-12)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_macro_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    test_auroc = roc_auc_score(y_true_test, y_prob_test, multi_class='ovr', average='macro')
    per_class_f1 = f1_score(y_true_test, y_pred_test, average=None, zero_division=0)
    cls_report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4)
    conf_mat = confusion_matrix(y_true_test, y_pred_test)

    with open(test_summary_csv, "w") as f:
        f.write("model,label_percent,test_loss,test_acc,test_macro_f1,test_auroc,f1_normal,f1_npdr,f1_pdr,best_val_f1\n")
        f.write(
            f"Multimodal_SSL,{label_percent},{avg_test_loss:.4f},{test_acc:.4f},{test_macro_f1:.4f},{test_auroc:.4f},"
            f"{per_class_f1[0]:.4f},{per_class_f1[1]:.4f},{per_class_f1[2]:.4f},{best_val_macro_f1:.4f}\n"
        )

    report_text = (
        f"============================================================\n"
        f"MULTIMODAL SSL FINE-TUNING ({label_percent}% LABELS) - FINAL TEST RESULTS\n"
        f"============================================================\n"
        f"Best Validation Macro-F1 : {best_val_macro_f1:.4f}\n"
        f"Test Loss                : {avg_test_loss:.4f}\n"
        f"Test Accuracy            : {test_acc:.4f}\n"
        f"Test Macro-F1            : {test_macro_f1:.4f}\n"
        f"Test AUROC (Macro OVR)   : {test_auroc:.4f}\n\n"
        f"Detailed Per-Class F1-Scores:\n"
        f"  - Normal : {per_class_f1[0]:.4f}\n"
        f"  - NPDR   : {per_class_f1[1]:.4f}\n"
        f"  - PDR    : {per_class_f1[2]:.4f}\n\n"
        f"Classification Report:\n{cls_report}\n"
        f"Confusion Matrix:\n{conf_mat}\n"
        f"============================================================\n"
    )
    print(report_text)
    with open(test_report_txt, "w") as f:
        f.write(report_text)
    print(f"📁 All fine-tuning logs and models saved to: {log_dir}")

if __name__ == "__main__":
    main()