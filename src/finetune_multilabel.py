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
from utils.metrics import calculate_metrics

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_pos_weights(dataset, train_indices):
    target_cols = ['diabetic_retinopathy', 'increased_cup_disc', 'macular_edema', 'myopic_fundus', 'amd']
    train_labels = dataset.df.iloc[train_indices][target_cols].values.astype(float)
    
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)
    soft_weights = np.power(raw_weights, 0.3)
    
    return torch.tensor(soft_weights, dtype=torch.float32)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multilabel Fine-tuning (5 Diseases) on {device}")

    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BRSETMultimodalDataset(
        csv_file=f"../{config['dataset']['csv_path']}",
        img_dir=f"../{config['dataset']['img_dir']}",
        transform=transform,
        task_type="multilabel_5"
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

    print("Computing smoothed class imbalance weights...")
    pos_weights = compute_pos_weights(dataset, train_dataset.indices).to(device)

    model = BRSETClassifier(config, task_name="multilabel_5").to(device)
    
    ssl_checkpoint_path = os.path.join(f"../{config['training']['checkpoint_dir']}", "best_model.pth")
    model.load_ssl_weights(ssl_checkpoint_path, device)

    # CRITICAL CHANGE: Gradual Unfreezing via Parameter Groups
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Two-speed optimizer: Very slow for backbone, normal for head
    optimizer = AdamW([
        {'params': ssl_params, 'lr': 1e-6},
        {'params': head_params, 'lr': config['finetuning']['learning_rate']}
    ], weight_decay=config['finetuning']['weight_decay'])

    log_dir = f"../{config['finetuning']['save_dir']}"
    os.makedirs(log_dir, exist_ok=True)
    history_csv = os.path.join(log_dir, "multilabel_history.csv")
    
    with open(history_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,auroc,macro_f1,dr_bacc,glaucoma_bacc,dme_bacc,myopia_bacc,amd_bacc\n")
        
    best_val_loss = float('inf')
    patience = 7
    epochs_no_improve = 0
    num_epochs = config['finetuning']['epochs']
    disease_names = ["Diabetic Retinopathy", "Glaucoma (Cup/Disc)", "Macular Edema", "Pathological Myopia", "AMD"]
    
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
            labels = batch['labels'].to(device).float()
            
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
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=False)
                tabular = batch['tabular'].to(device, non_blocking=False)
                masks = batch['mask'].to(device, non_blocking=False)
                labels = batch['labels'].to(device).float()
                
                logits = model(images, tabular, masks)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu())
                all_preds.append(probs.cpu())
                
        avg_val_loss = val_loss / len(val_loader)
        y_true = torch.cat(all_labels, dim=0)
        y_pred = torch.cat(all_preds, dim=0)
        
        metrics = calculate_metrics(y_true, y_pred, task_type="multilabel")
        b_accs = metrics['per_class_balanced_acc']
        
        with open(history_csv, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{metrics['auroc']:.4f},{metrics['macro_f1']:.4f},"
                    f"{b_accs[0]:.4f},{b_accs[1]:.4f},{b_accs[2]:.4f},{b_accs[3]:.4f},{b_accs[4]:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(log_dir, "best_multilabel_model.pth")
            torch.save(model.state_dict(), save_path)
            best_model_msg = f" | Best Model Saved (Val Loss: {best_val_loss:.4f})"
            epochs_no_improve = 0
        else:
            best_model_msg = ""
            epochs_no_improve += 1

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_model_msg}")
        print(f"Overall AUROC: {metrics['auroc']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
        
        opt_thresh = metrics.get('optimal_thresholds', [0.5]*5)
        print(f"Optimal Thresholds applied: {[round(t, 3) for t in opt_thresh]}")
        
        print("-" * 50)
        print("Val Set Balanced Accuracy Per Disease:")
        for name, b_acc in zip(disease_names, b_accs):
            print(f"  - {name}: {b_acc:.4f}")
        print("-" * 50)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement in Validation Loss for {patience} consecutive epochs.")
            break

    # ==========================================
    # FINAL TEST EVALUATION PHASE
    # ==========================================
    print("\n" + "="*60)
    print("Training Complete. Evaluating BEST model on TEST set...")
    print("="*60)
    
    best_model_path = os.path.join(log_dir, "best_multilabel_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model checkpoint not found. Using current weights.")

    model.eval()
    test_loss = 0.0
    all_labels_test = []
    all_preds_test = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch['image'].to(device, non_blocking=False)
            tabular = batch['tabular'].to(device, non_blocking=False)
            masks = batch['mask'].to(device, non_blocking=False)
            labels = batch['labels'].to(device).float()
            
            logits = model(images, tabular, masks)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_labels_test.append(labels.cpu())
            all_preds_test.append(probs.cpu())
            
    avg_test_loss = test_loss / len(test_loader)
    y_true_test = torch.cat(all_labels_test, dim=0)
    y_pred_test = torch.cat(all_preds_test, dim=0)
    
    test_metrics = calculate_metrics(y_true_test, y_pred_test, task_type="multilabel")
    test_b_accs = test_metrics['per_class_balanced_acc']
    
    print(f"\nFINAL TEST RESULTS | Test Loss: {avg_test_loss:.4f}")
    print(f"Overall AUROC: {test_metrics['auroc']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    
    opt_thresh_test = test_metrics.get('optimal_thresholds', [0.5]*5)
    print(f"Optimal Thresholds applied (Test): {[round(t, 3) for t in opt_thresh_test]}")
    
    print("-" * 50)
    print("Test Set Balanced Accuracy Per Disease (Compare with MetaFusion):")
    for name, b_acc in zip(disease_names, test_b_accs):
        print(f"  - {name}: {b_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()