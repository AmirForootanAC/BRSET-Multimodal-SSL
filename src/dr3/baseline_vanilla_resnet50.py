import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

from dataset import BRSETMultimodalDataset

def load_config():
    with open(os.path.join(CURRENT_DIR, "config_dr3.yaml"), "r") as f:
        return yaml.safe_load(f)

def clean_path(p):
    p = str(p)
    while "../" in p:
        p = p.replace("../", "")
    return p.strip("/")

def main():
    torch.backends.cudnn.benchmark = True
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Vanilla ResNet-50 (Image-Only, ImageNet Weights) Baseline on {device}")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))
    img_size = int(config['dataset'].get('finetune_image_size', 448))

    baseline_cfg = config.get('baseline', {})
    batch_size = int(baseline_cfg.get('batch_size', 16))
    grad_accum_steps = int(baseline_cfg.get('grad_accum_steps', 2))
    num_epochs = int(baseline_cfg.get('epochs', 40))
    patience = int(baseline_cfg.get('early_stopping_patience', 10))
    lr = float(baseline_cfg.get('learning_rate', 1e-4))
    weight_decay = float(baseline_cfg.get('weight_decay', 0.01))
    num_workers = int(baseline_cfg.get('num_workers', 4))

    save_dir = os.path.join(PROJECT_ROOT, clean_path(baseline_cfg.get('save_dir', 'data/processed/experiments_dr3_v2/baseline_vanilla_resnet50')))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_vanilla_baseline.pth")
    history_csv = os.path.join(save_dir, "baseline_history.csv")
    test_summary_csv = os.path.join(save_dir, "baseline_test_summary.csv")
    test_report_txt = os.path.join(save_dir, "baseline_test_report.txt")

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

    train_idx, temp_idx, _, temp_labels = train_test_split(indices, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)

    train_ds = Subset(BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=train_transform, task_type="dr_3class", is_train=True), train_idx)
    val_ds = Subset(BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False), val_idx)
    test_ds = Subset(BRSETMultimodalDataset(csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False), test_idx)

    class_counts = np.bincount(all_labels[train_idx])
    class_weights = torch.tensor(len(train_idx) / (3.0 * class_counts), dtype=torch.float32).to(device)
    print(f"📊 Split: {len(train_idx)} Train | {len(val_idx)} Val | {len(test_idx)} Test")
    print(f"⚖️ Class Counts: {class_counts.tolist()} | Class Weights: {[round(w, 3) for w in class_weights.tolist()]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    with open(history_csv, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_acc,val_macro_f1,val_auroc,val_f1_normal,val_f1_npdr,val_f1_pdr,is_best\n")

    best_val_f1 = 0.0
    epochs_no_improve = 0
    class_names = ["Normal", "NPDR", "PDR"]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).long()

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * grad_accum_steps
            train_loss += batch_loss
            pbar.set_postfix({'loss': f"{batch_loss:.4f}"})

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_y, all_probs = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val  ]", leave=False):
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True).long()
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    v_loss = criterion(logits, labels)
                val_loss += v_loss.item()
                probs = torch.softmax(logits.float(), dim=1).cpu()
                all_y.append(labels.cpu())
                all_probs.append(probs)

        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_y).numpy()
        y_prob = torch.cat(all_probs).numpy()
        row_sums = y_prob.sum(axis=1, keepdims=True)
        y_prob = y_prob / np.maximum(row_sums, 1e-12)
        y_pred = np.argmax(y_prob, axis=1)

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        per_class_val = f1_score(y_true, y_pred, average=None, zero_division=0)

        is_best = macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            best_marker = " 🌟 [Best Saved]"
        else:
            epochs_no_improve += 1
            best_marker = ""

        with open(history_csv, "a") as f:
            f.write(
                f"{epoch+1},{current_lr:.6e},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                f"{acc:.4f},{macro_f1:.4f},{auroc:.4f},"
                f"{per_class_val[0]:.4f},{per_class_val[1]:.4f},{per_class_val[2]:.4f},{int(is_best)}\n"
            )

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {macro_f1:.4f} (N:{per_class_val[0]:.2f}, NP:{per_class_val[1]:.2f}, P:{per_class_val[2]:.2f}) | "
            f"Acc: {acc:.4f} | AUROC: {auroc:.4f}{best_marker}"
        )

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}! No improvement for {patience} epochs.")
            break

    print("\n" + "="*60 + "\nEvaluating Best Vanilla Model on Test Set...\n" + "="*60)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_loss = 0.0
    test_y, test_probs = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).long()
            with torch.amp.autocast('cuda'):
                logits = model(images)
                t_loss = criterion(logits, labels)
            test_loss += t_loss.item()
            test_y.append(labels.cpu())
            test_probs.append(torch.softmax(logits.float(), dim=1).cpu())

    avg_test_loss = test_loss / len(test_loader)
    y_true_t = torch.cat(test_y).numpy()
    y_prob_t = torch.cat(test_probs).numpy()
    row_sums_t = y_prob_t.sum(axis=1, keepdims=True)
    y_prob_t = y_prob_t / np.maximum(row_sums_t, 1e-12)
    y_pred_t = np.argmax(y_prob_t, axis=1)

    test_acc = accuracy_score(y_true_t, y_pred_t)
    test_macro_f1 = f1_score(y_true_t, y_pred_t, average='macro')
    test_auroc = roc_auc_score(y_true_t, y_prob_t, multi_class='ovr', average='macro')
    per_class_t = f1_score(y_true_t, y_pred_t, average=None, zero_division=0)
    cls_report = classification_report(y_true_t, y_pred_t, target_names=class_names, digits=4)
    conf_mat = confusion_matrix(y_true_t, y_pred_t)

    with open(test_summary_csv, "w") as f:
        f.write("model,test_loss,test_acc,test_macro_f1,test_auroc,f1_normal,f1_npdr,f1_pdr,best_val_f1\n")
        f.write(
            f"Vanilla_ResNet50,{avg_test_loss:.4f},{test_acc:.4f},{test_macro_f1:.4f},{test_auroc:.4f},"
            f"{per_class_t[0]:.4f},{per_class_t[1]:.4f},{per_class_t[2]:.4f},{best_val_f1:.4f}\n"
        )

    report_text = (
        f"============================================================\n"
        f"VANILLA RESNET-50 BASELINE - FINAL TEST RESULTS\n"
        f"============================================================\n"
        f"Best Validation Macro-F1 : {best_val_f1:.4f}\n"
        f"Test Loss                : {avg_test_loss:.4f}\n"
        f"Test Accuracy            : {test_acc:.4f}\n"
        f"Test Macro-F1            : {test_macro_f1:.4f}\n"
        f"Test AUROC (Macro OVR)   : {test_auroc:.4f}\n\n"
        f"Per-Class F1 Scores:\n"
        f"  - Normal : {per_class_t[0]:.4f}\n"
        f"  - NPDR   : {per_class_t[1]:.4f}\n"
        f"  - PDR    : {per_class_t[2]:.4f}\n\n"
        f"Classification Report:\n{cls_report}\n"
        f"Confusion Matrix:\n{conf_mat}\n"
        f"============================================================\n"
    )
    print(report_text)
    with open(test_report_txt, "w") as f:
        f.write(report_text)
    print(f"📁 All baseline logs and models saved to: {save_dir}")

if __name__ == "__main__":
    main()