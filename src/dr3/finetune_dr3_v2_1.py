import os
import sys
import copy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_config(config_path="config_dr3_v2_1.yaml"):
    full_path = os.path.join(CURRENT_DIR, config_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(CURRENT_DIR, "config_dr3_v2.yaml")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_path(path_str):
    path_str = str(path_str)
    if os.path.isabs(path_str):
        return path_str
    while "../" in path_str:
        path_str = path_str.replace("../", "")
    return path_str.strip("/")


class ModelEMA:
    def __init__(self, model, decay=0.992):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


class CalibratedFocalWeightedLoss(nn.Module):
    """
    Combines Calibrated Class Weighting with a mild Focal modulation (gamma=1.25)
    so hard border-line cases (NPDR vs Normal) receive stronger gradients without
    distorting global probability calibration (preserving ~0.97 AUROC).
    """
    def __init__(self, class_weights, gamma=1.25, smoothing=0.03, logit_reg=0.003):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.smoothing = smoothing
        self.logit_reg = logit_reg

    def forward(self, logits, targets):
        logits_fp32 = logits.float()
        log_probs = F.log_softmax(logits_fp32, dim=-1)
        probs = torch.exp(log_probs)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - pt).pow(self.gamma)

        per_sample_nll = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction='none')
        focal_nll = (focal_factor * per_sample_nll).mean()

        smooth_loss = -log_probs.mean(dim=-1).mean()
        logit_penalty = (logits_fp32.pow(2).mean()) * self.logit_reg

        return (1.0 - self.smoothing) * focal_nll + self.smoothing * smooth_loss + logit_penalty


def set_backbone_trainable(model, trainable=True):
    for name, param in model.named_parameters():
        if 'ssl_base' in name:
            param.requires_grad = trainable


def optimize_class_thresholds(y_true, y_prob):
    """
    Finds optimal probability multipliers [1.0, m_npdr, m_pdr] on the Validation set
    to maximize Macro-F1 without changing raw probabilities (so AUROC is 100% untouched).
    """
    best_f1 = f1_score(y_true, np.argmax(y_prob, axis=1), average='macro')
    best_mult = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for m1 in np.linspace(0.85, 2.40, 32):
        for m2 in np.linspace(0.85, 2.00, 24):
            mult = np.array([1.0, m1, m2], dtype=np.float32)
            adj_pred = np.argmax(y_prob * mult, axis=1)
            score = f1_score(y_true, adj_pred, average='macro')
            if score > best_f1:
                best_f1 = score
                best_mult = mult
    return best_mult, best_f1


def evaluate_model(eval_model, loader, criterion, device, use_amp, desc="Evaluating", leave=False, class_mult=None):
    eval_model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    progress_bar = tqdm(loader, desc=desc, leave=leave)
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).long()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = eval_model(images, tabular, masks)
                loss = criterion(logits, labels)

            batch_loss = loss.item()
            total_loss += batch_loss
            probs = torch.softmax(logits.float(), dim=1).cpu()
            all_labels.append(labels.cpu())
            all_probs.append(probs)
            progress_bar.set_postfix({'loss': f"{batch_loss:.4f}"})

    avg_loss = total_loss / max(len(loader), 1)
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()

    row_sums = y_prob.sum(axis=1, keepdims=True)
    y_prob = y_prob / np.maximum(row_sums, 1e-12)

    if class_mult is not None:
        y_pred = np.argmax(y_prob * class_mult, axis=1)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    return avg_loss, acc, macro_f1, auroc, per_class_f1, y_true, y_pred, y_prob


def main():
    torch.backends.cudnn.benchmark = True
    config = load_config("config_dr3_v2_1.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_percent = float(config['finetuning'].get('label_percent', 100.0))
    label_fraction = label_percent / 100.0
    img_size = int(config['dataset'].get('finetune_image_size', 448))
    batch_size = int(config['finetuning']['batch_size'])
    grad_accum_steps = int(config['finetuning'].get('grad_accum_steps', 2))
    num_workers = int(config['finetuning'].get('num_workers', 4))
    modality_dropout_p = float(config['fusion'].get('modality_dropout_p', 0.20))
    warmup_freeze_epochs = int(config['finetuning'].get('warmup_freeze_epochs', 2))
    weight_power = float(config['finetuning'].get('class_weight_power', 0.55))
    fusion_type = str(config['fusion'].get('fusion_type', 'cross_attention'))

    print(f"🚀 Starting 3-Class Pure Multimodal SSL Fine-tuning V2.1 (Fusion: {fusion_type}) on {device}")
    print(f"📊 Label Efficiency Mode: Using {label_percent}% of training data | Warmup Freeze Epochs: {warmup_freeze_epochs}")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))
    vision_weights = os.path.join(PROJECT_ROOT, clean_path(config['model']['vision_weights']))

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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

    base_ds = BRSETMultimodalDataset(
        csv_file=csv_path, img_dir=img_dir, transform=val_test_transform, task_type="dr_3class", is_train=False
    )
    all_labels = base_ds.labels
    indices = np.arange(len(base_ds))

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )

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
        BRSETMultimodalDataset(
            csv_file=csv_path, img_dir=img_dir, transform=train_transform,
            task_type="dr_3class", is_train=True, modality_dropout_p=modality_dropout_p
        ),
        active_train_idx
    )
    val_dataset = Subset(
        BRSETMultimodalDataset(
            csv_file=csv_path, img_dir=img_dir, transform=val_test_transform,
            task_type="dr_3class", is_train=False, modality_dropout_p=0.0
        ),
        val_idx
    )
    test_dataset = Subset(
        BRSETMultimodalDataset(
            csv_file=csv_path, img_dir=img_dir, transform=val_test_transform,
            task_type="dr_3class", is_train=False, modality_dropout_p=0.0
        ),
        test_idx
    )

    class_counts = np.bincount(all_labels[active_train_idx])
    raw_weights = len(active_train_idx) / (3.0 * np.maximum(class_counts, 1))
    smoothed_weights = np.power(raw_weights, weight_power)
    class_weights = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    print(f"📊 Dataset split: {len(active_train_idx)} Train | {len(val_idx)} Val | {len(test_idx)} Test")
    print(f"⚖️ Class Counts: {class_counts.tolist()} | Calibrated Weights (p={weight_power}): {[round(w, 3) for w in class_weights.tolist()]}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )

    model = BRSETClassifier(config, task_name="dr_3class", vision_weights_path=vision_weights).to(device)

    ssl_checkpoint_path = os.path.join(
        PROJECT_ROOT, clean_path(config['ssl_training']['checkpoint_dir']), "best_model.pth"
    )
    model.load_ssl_weights(ssl_checkpoint_path, device)

    ssl_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ssl_base' in name:
            ssl_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = float(config['finetuning'].get('backbone_lr', 3e-5))
    head_lr = float(config['finetuning'].get('learning_rate', 3e-4))
    num_epochs = int(config['finetuning'].get('epochs', 50))
    focal_gamma = float(config['finetuning'].get('focal_gamma', 1.25))

    criterion = CalibratedFocalWeightedLoss(
        class_weights=class_weights, gamma=focal_gamma, smoothing=0.03, logit_reg=0.003
    )
    optimizer = AdamW([
        {'params': ssl_params, 'lr': backbone_lr, 'weight_decay': 0.01},
        {'params': head_params, 'lr': head_lr, 'weight_decay': 0.01}
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    use_amp = bool(config['finetuning'].get('mixed_precision', True))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ema_helper = ModelEMA(model, decay=0.992)

    base_save_dir = os.path.join(PROJECT_ROOT, clean_path(config['finetuning']['save_dir']))
    percent_str = str(int(label_percent)) if label_percent.is_integer() else str(label_percent)
    log_dir = os.path.join(base_save_dir, f"dr3_percent_{percent_str}")
    os.makedirs(log_dir, exist_ok=True)

    history_csv = os.path.join(log_dir, f"finetune_history_v2_1_percent_{percent_str}.csv")
    save_path = os.path.join(log_dir, f"best_model_v2_1_percent_{percent_str}.pth")
    test_summary_csv = os.path.join(log_dir, f"finetune_test_summary_v2_1_percent_{percent_str}.csv")
    test_report_txt = os.path.join(log_dir, f"finetune_test_report_v2_1_percent_{percent_str}.txt")

    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,lr_backbone,lr_head,train_loss,val_loss,val_acc,val_macro_f1,val_auroc,composite_score,val_f1_normal,val_f1_npdr,val_f1_pdr,is_best\n")

    best_composite_score = 0.0
    best_val_macro_f1 = 0.0
    best_val_auroc = 0.0
    patience = int(config['finetuning'].get('early_stopping_patience', 15))
    epochs_no_improve = 0
    class_names = ["Normal", "NPDR", "PDR"]

    for epoch in range(num_epochs):
        if epoch < warmup_freeze_epochs:
            set_backbone_trainable(model, trainable=False)
        elif epoch == warmup_freeze_epochs:
            set_backbone_trainable(model, trainable=True)
            print(f"🔓 Epoch {epoch+1}: Unfreezing SSL Backbone with gentle LR ({backbone_lr:.1e})")

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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_helper.update(model)

            batch_loss = loss.item() * grad_accum_steps
            train_loss += batch_loss
            train_progress_bar.set_postfix({'loss': f"{batch_loss:.4f}"})

        cur_lr_bb = 0.0 if epoch < warmup_freeze_epochs else optimizer.param_groups[0]['lr']
        cur_lr_hd = optimizer.param_groups[1]['lr']
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        val_loss_raw, acc_raw, f1_raw, auroc_raw, per_cls_raw, _, _, _ = evaluate_model(
            model, val_loader, criterion, device, use_amp,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val-RAW]", leave=False
        )
        val_loss_ema, acc_ema, f1_ema, auroc_ema, per_cls_ema, _, _, _ = evaluate_model(
            ema_helper.ema_model, val_loader, criterion, device, use_amp,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val-EMA]", leave=False
        )

        score_raw = 0.72 * f1_raw + 0.28 * auroc_raw
        score_ema = 0.72 * f1_ema + 0.28 * auroc_ema

        if score_ema >= score_raw:
            avg_val_loss, acc, macro_f1, auroc, per_class_val = val_loss_ema, acc_ema, f1_ema, auroc_ema, per_cls_ema
            composite_score = score_ema
            active_state_dict = ema_helper.ema_model.state_dict()
            src_tag = "EMA"
        else:
            avg_val_loss, acc, macro_f1, auroc, per_class_val = val_loss_raw, acc_raw, f1_raw, auroc_raw, per_cls_raw
            composite_score = score_raw
            active_state_dict = model.state_dict()
            src_tag = "RAW"

        is_best = composite_score > best_composite_score
        if is_best:
            best_composite_score = composite_score
            best_val_macro_f1 = macro_f1
            best_val_auroc = auroc
            torch.save(active_state_dict, save_path)
            status_msg = f" 🌟 [Best Saved ({src_tag}) | F1:{macro_f1:.4f}, AUC:{auroc:.4f}]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            status_msg = f" ⚠️ [No Improve: {epochs_no_improve}/{patience}]"

        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{cur_lr_bb:.6e},{cur_lr_hd:.6e},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                f"{acc:.4f},{macro_f1:.4f},{auroc:.4f},{composite_score:.4f},"
                f"{per_class_val[0]:.4f},{per_class_val[1]:.4f},{per_class_val[2]:.4f},{int(is_best)}\n"
            )

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
            f"Val F1: {macro_f1:.4f} (N:{per_class_val[0]:.2f}, NP:{per_class_val[1]:.2f}, P:{per_class_val[2]:.2f}) | "
            f"Acc: {acc:.4f} | AUROC: {auroc:.4f}{status_msg}"
        )

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! No improvement in Composite Score for {patience} epochs.")
            break

    print("\n" + "="*60)
    print("Training Complete. Evaluating BEST V2.1 Model on TEST set...")
    print("="*60)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best calibrated checkpoint from {save_path}")

    # Step 1: Standard (argmax) evaluation on Validation to find calibrated decision thresholds
    _, _, val_f1_std, val_auc_std, _, y_true_val, _, y_prob_val = evaluate_model(
        model, val_loader, criterion, device, use_amp,
        desc="Calibrating Thresholds on Val Set", leave=False
    )
    best_mult, val_f1_cal = optimize_class_thresholds(y_true_val, y_prob_val)
    print(f"🎯 Post-Hoc Threshold Calibration (Val): Standard F1={val_f1_std:.4f} -> Calibrated F1={val_f1_cal:.4f} | Multipliers={np.round(best_mult, 3).tolist()}")

    # Step 2: Evaluate on Test Set (both Standard Argmax and Threshold-Calibrated)
    avg_test_loss, test_acc_std, test_f1_std, test_auroc, per_class_std, y_true_test, y_pred_std, y_prob_test = evaluate_model(
        model, test_loader, criterion, device, use_amp,
        desc="Evaluating Test Set", leave=True
    )

    y_pred_cal = np.argmax(y_prob_test * best_mult, axis=1)
    test_acc_cal = accuracy_score(y_true_test, y_pred_cal)
    test_f1_cal = f1_score(y_true_test, y_pred_cal, average='macro')
    per_class_cal = f1_score(y_true_test, y_pred_cal, average=None, zero_division=0)

    # Use whichever gives better generalization or report both clearly
    cls_report_std = classification_report(y_true_test, y_pred_std, target_names=class_names, digits=4)
    conf_mat_std = confusion_matrix(y_true_test, y_pred_std)

    cls_report_cal = classification_report(y_true_test, y_pred_cal, target_names=class_names, digits=4)
    conf_mat_cal = confusion_matrix(y_true_test, y_pred_cal)

    with open(test_summary_csv, "w", encoding="utf-8") as f:
        f.write("model,eval_mode,label_percent,test_loss,test_acc,test_macro_f1,test_auroc,f1_normal,f1_npdr,f1_pdr,best_val_f1,best_val_auroc\n")
        f.write(
            f"Multimodal_Pure_SSL_V2.1,Standard_Argmax,{label_percent},{avg_test_loss:.4f},{test_acc_std:.4f},{test_f1_std:.4f},{test_auroc:.4f},"
            f"{per_class_std[0]:.4f},{per_class_std[1]:.4f},{per_class_std[2]:.4f},{best_val_macro_f1:.4f},{best_val_auroc:.4f}\n"
        )
        f.write(
            f"Multimodal_Pure_SSL_V2.1,Calibrated_Threshold,{label_percent},{avg_test_loss:.4f},{test_acc_cal:.4f},{test_f1_cal:.4f},{test_auroc:.4f},"
            f"{per_class_cal[0]:.4f},{per_class_cal[1]:.4f},{per_class_cal[2]:.4f},{val_f1_cal:.4f},{best_val_auroc:.4f}\n"
        )

    report_text = (
        f"============================================================\n"
        f"PURE MULTIMODAL SSL V2.1 (CROSS-ATTENTION) TEST RESULTS ({label_percent}% LABELS)\n"
        f"============================================================\n"
        f"Best Validation Macro-F1 (Standard)   : {best_val_macro_f1:.4f}\n"
        f"Best Validation Macro-F1 (Calibrated) : {val_f1_cal:.4f}\n"
        f"Best Validation AUROC                 : {best_val_auroc:.4f}\n"
        f"Test Loss                             : {avg_test_loss:.4f}\n"
        f"Test AUROC (Macro OVR - Identical)    : {test_auroc:.4f}\n"
        f"------------------------------------------------------------\n"
        f"[A] STANDARD ARGMAX EVALUATION:\n"
        f"Test Accuracy : {test_acc_std:.4f} | Test Macro-F1 : {test_f1_std:.4f}\n"
        f"Per-Class F1  : Normal={per_class_std[0]:.4f}, NPDR={per_class_std[1]:.4f}, PDR={per_class_std[2]:.4f}\n\n"
        f"Classification Report (Standard):\n{cls_report_std}\n"
        f"Confusion Matrix (Standard):\n{conf_mat_std}\n"
        f"------------------------------------------------------------\n"
        f"[B] VAL-CALIBRATED THRESHOLD EVALUATION (Multipliers={np.round(best_mult, 3).tolist()}):\n"
        f"Test Accuracy : {test_acc_cal:.4f} | Test Macro-F1 : {test_f1_cal:.4f}\n"
        f"Per-Class F1  : Normal={per_class_cal[0]:.4f}, NPDR={per_class_cal[1]:.4f}, PDR={per_class_cal[2]:.4f}\n\n"
        f"Classification Report (Calibrated):\n{cls_report_cal}\n"
        f"Confusion Matrix (Calibrated):\n{conf_mat_cal}\n"
        f"============================================================\n"
    )
    print(report_text)
    with open(test_report_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📁 All fine-tuning V2.1 logs and models saved to: {log_dir}")


if __name__ == "__main__":
    main()
