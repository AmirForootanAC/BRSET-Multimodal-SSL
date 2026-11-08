import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, roc_curve

def calculate_metrics(y_true, y_pred_probs, task_type="multilabel"):
    y_true_np = y_true.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()
    
    metrics = {}
    
    try:
        if task_type == "multilabel":
            metrics['auroc'] = roc_auc_score(y_true_np, y_pred_probs_np, average='macro', multi_class='ovr')
            
            optimal_thresholds = []
            balanced_accs = []
            y_pred_binary = np.zeros_like(y_pred_probs_np)
            
            # Find optimal threshold per class using Youden's J statistic
            for i in range(y_true_np.shape[1]):
                fpr, tpr, thresholds = roc_curve(y_true_np[:, i], y_pred_probs_np[:, i])
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_thresh = thresholds[best_idx]
                optimal_thresholds.append(best_thresh)
                
                y_pred_binary[:, i] = (y_pred_probs_np[:, i] >= best_thresh).astype(int)
                b_acc = balanced_accuracy_score(y_true_np[:, i], y_pred_binary[:, i])
                balanced_accs.append(b_acc)
            
            metrics['accuracy'] = accuracy_score(y_true_np, y_pred_binary)
            metrics['macro_f1'] = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
            metrics['per_class_balanced_acc'] = balanced_accs
            metrics['macro_balanced_acc'] = np.mean(balanced_accs)
            metrics['optimal_thresholds'] = optimal_thresholds
            
        elif task_type == "multiclass":
            y_pred_classes = np.argmax(y_pred_probs_np, axis=1)
            metrics['accuracy'] = accuracy_score(y_true_np, y_pred_classes)
            metrics['macro_f1'] = f1_score(y_true_np, y_pred_classes, average='macro', zero_division=0)
            metrics['macro_balanced_acc'] = balanced_accuracy_score(y_true_np, y_pred_classes)
            
            num_classes = y_pred_probs_np.shape[1]
            y_true_onehot = np.eye(num_classes)[y_true_np]
            metrics['auroc'] = roc_auc_score(y_true_onehot, y_pred_probs_np, average='macro', multi_class='ovr')
            
        elif task_type == "binary":
            fpr, tpr, thresholds = roc_curve(y_true_np, y_pred_probs_np)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            
            y_pred_binary = (y_pred_probs_np >= best_thresh).astype(int)
            metrics['accuracy'] = accuracy_score(y_true_np, y_pred_binary)
            metrics['f1'] = f1_score(y_true_np, y_pred_binary, zero_division=0)
            metrics['auroc'] = roc_auc_score(y_true_np, y_pred_probs_np)
            metrics['balanced_acc'] = balanced_accuracy_score(y_true_np, y_pred_binary)
            
    except ValueError as e:
        print(f"Warning in metric calculation: {e}")
        metrics['auroc'] = 0.0
        
    return metrics