import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


    

def compute_supervised_metrics(metrics, logits, labels, suffix):
    result = {}
    _, preds = torch.max(logits, 1)
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
        
    
    return result

