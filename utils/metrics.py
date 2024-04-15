import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def accuracy(logits, label):
    pred = logits.max(1)[1]
    correct = (pred == label).sum()
    total = logits.shape[0]
    return correct / total * 100


# def precision(logits, labels):
#     pred = logits.max(1)[1]
    
    
    
    
def precision_recall_f1(logits, labels):
    _, preds = torch.max(logits, 1)
    preds = preds.cpu().numpy()  # 转换为numpy数组
    labels = labels.cpu().numpy()  # 转换为numpy数组
    
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    return precision, recall, f1
