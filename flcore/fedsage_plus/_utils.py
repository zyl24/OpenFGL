import torch
import numpy as np
import torch.nn.functional as F

def greedy_loss(pred_feats, true_feats, pred_missing, true_missing, max_pred, device):
    true_missing = true_missing.to(device)
    pred_missing = pred_missing.to(device)
    loss=torch.zeros(pred_feats.shape)
    loss=loss.to(device)
    
    
    pred_len=len(pred_feats)
    pred_missing_np = pred_missing.detach().numpy().reshape(-1).astype(np.int32)
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(np.int32)
    true_missing_np = np.clip(true_missing_np,0, max_pred)
    pred_missing_np = np.clip(pred_missing_np, 0, max_pred)
    for i in range(pred_len):
        for pred_j in range(min(max_pred, pred_missing_np[i])):
            if true_missing_np[i]>0:
                if isinstance(true_feats[i][true_missing_np[i]-1], np.ndarray):
                    true_feats_tensor = torch.tensor(true_feats[i][true_missing_np[i]-1]).to(device)
                else:
                    true_feats_tensor=true_feats[i][true_missing_np[i]-1]
                loss[i][pred_j] += F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                                  true_feats_tensor.unsqueeze(0).float()).squeeze(0)

                for true_k in range(min(max_pred, true_missing_np[i])):
                    if isinstance(true_feats[i][true_k], np.ndarray):
                        true_feats_tensor = torch.tensor(true_feats[i][true_k]).to(device)
                    else:
                        true_feats_tensor = true_feats[i][true_k]

                    loss_ijk = F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                        true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if torch.sum(loss_ijk)<torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j]=loss_ijk
            else:
                continue
    return loss
