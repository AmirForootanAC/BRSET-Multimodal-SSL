import os
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

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_class_weights(dataset, train_indices):
    """
    Computes class weights inversely proportional to class frequencies 
    (FAST VERSION - Reads directly from DataFrame memory)
    Classes: 0 (Normal), 1 (NPDR), 2 (PDR)
    """
    print("Computing class weights to combat severe Normal vs PDR imbalance (FAST mode)...")
    
    train_df = dataset.df.iloc[train_indices]
    dr_raw = train_df['DR_ICDR'].values
    
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for val in dr_raw:
        if val == 0:
            class_counts[0] += 1
        elif val in [1, 2]:
            class_counts[1] += 1
        else:
            class_counts[2] += 1
            
    counts_array = np.array([class_counts[0], class_counts[1], class_counts[2]])
    total = counts_array.sum()
    
    # Inverse frequency weighting with smoothing
    weights = total / (len(class_counts) * counts_array)
    smoothed_weights = np.power(weights, 0.5)
    
    print(f"Class Counts (0: Normal, 1: NPDR, 2: PDR): {counts_array}")
    print(f"Applied Class Weights: {[round(w, 3) for w in smoothed_weights]}")
    
    return torch.tensor(smoothed_weights, dtype=torch.float32)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting 3-Class DR Fine-tuning (Weighted CE + AUROC) on {device}")

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset configured for 3-class DR task
    dataset = BRSETMultimodalDataset(
        csv_file=f"../{config['dataset']['csv_path']}",
        img_dir=f"../{config['dataset']['img_dir']}",
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

    batch_size = config['finetuning']['batch_size']
    num_workers = config['training']['num_workers']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    print(f"Dataset split: {train_size} Train | {val_size} Val | {test_size} Test")

    # Compute optimal weights for the highly imbalanced classes
    class_weights = compute_class_weights(dataset, train_dataset.indices).to(device)

    # Initialize model using standard V1 initialization
    model = BRSETClassifier(config, task_name="dr_3class").to(device)
    
    ssl_checkpoint_path = os.path.join(f"../{config['training']['checkpoint_dir']}", "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    # Safely override the final classification head output features for 3-class DR task
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
        if not param.requires_grad:
            continue
        if 'ssl_base' in name:
            ssl_params.append(param)
        else:
            head_params.append(param)

    # Use Weighted CrossEntropyLoss to force model to learn PDR/NPDR
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW([
        {'params': ssl_params, 'lr': 1e-6},
        {'params': head_params, 'lr': config['finetuning']['learning_rate']}
    ], weight_decay=config['finetuning']['weight_decay'])

    log_dir = f"../{config['finetuning']['save_dir']}"
    os.makedirs(log_dir, exist_ok=True)
    history_csv = os.path.join(log_dir, "dr3_history_weighted.csv")
    
    with open(history_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,accuracy,macro_f1,auroc\n")
        
    best_val_macro_f1 = 0.0
    patience = 6
    epochs_no_improve = 0
    num_epochs = 15  # 15-Epoch Benchmark Test
    class_names = ["Normal", "NPDR", "PDR"]
    
    # ==========================================
    # TRAINING & VALIDATION LOOP
    # ==========================================
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=False)
            tabular = batch['tabular'].to(device, non_blocking=False)
            masks = batch['mask'].to(device, non_blocking=False)
            labels = batch['labels'].to(device).long()
            
            optimizer.zero_grad()
            logits = model(images, tabular, masks)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=False)
                tabular = batch['tabular'].to(device, non_blocking=False)
                masks = batch['mask'].to(device, non_blocking=False)
                labels = batch['labels'].to(device).long()
                
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        
        with open(history_csv, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f},{auroc:.4f}\n")

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            save_path = os.path.join(log_dir, "best_dr3_model_weighted.pth")
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

    # ==========================================
    # FINAL TEST EVALUATION PHASE
    # ==========================================
    print("\n" + "="*60)
    print("Training Complete. Evaluating BEST model on TEST set...")
    print("="*60)
    
    best_model_path = os.path.join(log_dir, "best_dr3_model_weighted.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model checkpoint not found. Using current weights.")

    model.eval()
    test_loss = 0.0
    all_labels_test = []
    all_preds_test = []
    all_probs_test = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(device, non_blocking=False)
            tabular = batch['tabular'].to(device, non_blocking=False)
            masks = batch['mask'].to(device, non_blocking=False)
            labels = batch['labels'].to(device).long()
            
            logits = model(images, tabular, masks)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels_test.append(labels.cpu())
            all_preds_test.append(preds.cpu())
            all_probs_test.append(probs.cpu())
            
    avg_test_loss = test_loss / len(test_loader)
    y_true_test = torch.cat(all_labels_test, dim=0).numpy()
    y_pred_test = torch.cat(all_preds_test, dim=0).numpy()
    y_prob_test = torch.cat(all_probs_test, dim=0).numpy()
    
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_macro_f1 = f1_score(y_true_test, y_pred_test, average='macro')
    test_auroc = roc_auc_score(y_true_test, y_prob_test, multi_class='ovr', average='macro')
    
    print(f"\nFINAL TEST RESULTS (3-Class DR Weighted) | Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Test AUROC: {test_auroc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f} (Baseline to beat: 0.82)")
    
    # Compute detailed per-class F1 to see exactly how PDR/NPDR are performing
    per_class_f1 = f1_score(y_true_test, y_pred_test, average=None)
    print("\nDetailed Per-Class F1-Scores:")
    for name, score in zip(class_names, per_class_f1):
        print(f"  - {name}: {score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()