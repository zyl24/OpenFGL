import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    

def compute_supervised_metrics(metrics, logits, labels, suffix):
    result = {}
    if logits.dim() == 1:
        probs = F.sigmoid(logits)
        preds = (probs > 0.5).long()
    else:
        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)
    
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    if "accuracy" in metrics:
        result[f"accuracy_{suffix}"] = accuracy_score(labels, preds)
    
    if "precision" in metrics:
        result[f"precision_{suffix}"] = precision_score(labels, preds, average='macro')

    if "recall" in metrics:
        result[f"recall_{suffix}"] = recall_score(labels, preds, average='macro')
        
    if "f1" in metrics:
        result[f"f1_{suffix}"] = f1_score(labels, preds, average='macro')
        
    if "auc" in metrics:
        if labels.max() > 1:
            raise ValueError("AUC is not directly supported for multi-class classification.")
        result[f"auc_{suffix}"] = roc_auc_score(labels, probs)
    
    if "roc" in metrics:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1] if probs.ndim > 1 else probs)
        result[f"roc_{suffix}"] = (fpr, tpr)

    if "ap" in metrics:
        result[f"ap_{suffix}"] = average_precision_score(labels, probs[:, 1] if probs.ndim > 1 else probs)
        
    return result

