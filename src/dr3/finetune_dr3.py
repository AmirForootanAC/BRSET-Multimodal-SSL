import os
import sys

# Dynamic path resolution to avoid relative path conflicts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from dataset import BRSETMultimodalDataset
from models.fusion import BRSETClassifier
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def load_config(config_path="config_dr3.yaml"):
    with open(os.path.join(CURRENT_DIR, config_path), "r") as f:
        return yaml.safe_load(f)

def clean_path(path_str):
    path_str = str(path_str)
    while "../" in path_str:
        path_str = path_str.replace("../", "")
    return path_str.strip("/")

def inject_compatibility(config):
    if 'finetuning' not in config:
        config['finetuning'] = {}
    if 'tasks' not in config['finetuning']:
        config['finetuning']['tasks'] = {}
        
    if 'task' in config:
        task_name = config['task'].get('name', 'dr_3class')
        config['finetuning']['tasks'][task_name] = config['task']
    return config

def compute_class_weights(dataset, train_indices):
    print("Computing class weights to combat severe imbalance (FAST mode)...")
    
    train_df = dataset.df.iloc[train_indices]
    dr_raw = train_df['DR_ICDR'].values
    
    class_counts = {0: 0, 1: 0, 2: 0}
    for val in dr_raw:
        if val == 0: class_counts[0] += 1
        elif val in [1, 2]: class_counts[1] += 1
        else: class_counts[2] += 1
            
    counts_array = np.array([class_counts[0], class_counts[1], class_counts[2]])
    counts_array = np.maximum(counts_array, 1) 
    total = counts_array.sum()
    
    weights = total / (len(class_counts) * counts_array)
    smoothed_weights = np.power(weights, 0.5)
    
    print(f"Class Counts (0: Normal, 1: NPDR, 2: PDR): {counts_array}")
    print(f"Applied Class Weights: {[round(w, 3) for w in smoothed_weights]}")
    
    return torch.tensor(smoothed_weights, dtype=torch.float32)

def main():
    config = load_config()
    config = inject_compatibility(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    label_percent = float(config['finetuning'].get('label_percent', 100.0))
    label_fraction = label_percent / 100.0
    
    print(f"🚀 Starting 3-Class DR Fine-tuning on {device}")
    print(f"📊 Label Efficiency Mode: Using {label_percent}% of training data")

    csv_path = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['csv_path']))
    img_dir = os.path.join(PROJECT_ROOT, clean_path(config['dataset']['img_dir']))

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transform=transform,
        task_type="dr_3class"
    )
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    if label_fraction < 1.0:
        train_labels_raw = dataset.df.iloc[train_dataset.indices]['DR_ICDR'].values
        train_labels_mapped = np.array([0 if x == 0 else (1 if x in [1, 2] else 2) for x in train_labels_raw])
        
        fractional_indices, _ = train_test_split(
            train_dataset.indices, 
            train_size=label_fraction, 
            random_state=42, 
            stratify=train_labels_mapped
        )
        train_dataset_fractional = torch.utils.data.Subset(dataset, fractional_indices)
        active_train_indices = fractional_indices
    else:
        train_dataset_fractional = train_dataset
        active_train_indices = train_dataset.indices

    print(f"Dataset split: {len(active_train_indices)} Train | {val_size} Val | {test_size} Test")

    batch_size = config['finetuning']['batch_size']
    num_workers = config['finetuning'].get('num_workers', 8)

    train_loader = DataLoader(train_dataset_fractional, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_weights = compute_class_weights(dataset, active_train_indices).to(device)

    model = BRSETClassifier(config, task_name="dr_3class").to(device)
    
    ssl_config = config.get('ssl_training', config.get('training', {}))
    ssl_checkpoint_path = os.path.join(PROJECT_ROOT, clean_path(ssl_config['checkpoint_dir']), "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 5:
            in_features = module.in_features
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            layer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name == '':
                setattr(model, layer_name, nn.Linear(in_features, 3))
            else:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, layer_name, nn.Linear(in_features, 3))
            break

    print("Unfreezing SSL backbone with a microscopic learning rate (1e-6)...")
    ssl_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'ssl_base' in name: ssl_params.append(param)
        else: head_params.append(param)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW([
        {'params': ssl_params, 'lr': 1e-6},
        {'params': head_params, 'lr': config['finetuning']['learning_rate']}
    ], weight_decay=config['finetuning']['weight_decay'])

    use_amp = config['finetuning'].get('mixed_precision', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    base_save_dir = os.path.join(PROJECT_ROOT, clean_path(config['finetuning']['save_dir']))
    percent_str = str(int(label_percent)) if label_percent.is_integer() else str(label_percent)
    
    run_folder_name = f"dr3_percent_{percent_str}"
    log_dir = os.path.join(base_save_dir, run_folder_name)
    os.makedirs(log_dir, exist_ok=True)
    
    history_csv = os.path.join(log_dir, f"history_percent_{percent_str}.csv")
    save_path = os.path.join(log_dir, f"best_model_percent_{percent_str}.pth")
    
    with open(history_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,accuracy,macro_f1,auroc\n")
        
    best_val_macro_f1 = 0.0
    patience = int(config['finetuning'].get('early_stopping_patience', 15))
    epochs_no_improve = 0
    num_epochs = int(config['finetuning'].get('epochs', 50))
    class_names = ["Normal", "NPDR", "PDR"]
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device).long()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []
        
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val  ]", leave=False)
        
        with torch.no_grad():
            for batch in val_progress_bar:
                images = batch['image'].to(device, non_blocking=True)
                tabular = batch['tabular'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device).long()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(images, tabular, masks)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1).float() 
                preds = torch.argmax(logits, dim=1)
                
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                
                val_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()
        
        # Ensure exact probability sum to 1.0 to prevent sklearn ValueError
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        
        with open(history_csv, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f},{auroc:.4f}\n")

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            best_model_msg = f" | 🌟 Best Model Saved (Macro-F1: {best_val_macro_f1:.4f})"
            epochs_no_improve = 0
        else:
            best_model_msg = ""
            epochs_no_improve += 1

        print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_model_msg}")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | AUROC: {auroc:.4f}")
        print("-" * 50)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement in Macro-F1 for {patience} consecutive epochs.")
            break

    print("\n" + "="*60)
    print("Training Complete. Evaluating BEST model on TEST set...")
    print("="*60)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded best model from {save_path}")

    model.eval()
    test_loss = 0.0
    all_labels_test, all_preds_test, all_probs_test = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device).long()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
                
            test_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).float()
            preds = torch.argmax(logits, dim=1)
            
            all_labels_test.append(labels.cpu())
            all_preds_test.append(preds.cpu())
            all_probs_test.append(probs.cpu())
            
    avg_test_loss = test_loss / len(test_loader)
    y_true_test = torch.cat(all_labels_test, dim=0).numpy()
    y_pred_test = torch.cat(all_preds_test, dim=0).numpy()
    y_prob_test = torch.cat(all_probs_test, dim=0).numpy()
    
    y_prob_test = y_prob_test / y_prob_test.sum(axis=1, keepdims=True)
    
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_macro_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    test_auroc = roc_auc_score(y_true_test, y_prob_test, multi_class='ovr', average='macro')
    
    print(f"\nFINAL TEST RESULTS (Fraction: {label_percent}%) | Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Test AUROC: {test_auroc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f} (Baseline to beat: 0.82)")
    
    per_class_f1 = f1_score(y_true_test, y_pred_test, average=None)
    print("\nDetailed Per-Class F1-Scores:")
    for name, score in zip(class_names, per_class_f1):
        print(f"  - {name}: {score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()