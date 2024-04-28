import os
import os.path as osp
from decimal import Decimal
import torch

def random_feature_mask(local_data, processed_dir, mask_prob=0.1):
    mask_root = osp.join(processed_dir, f"mask_{Decimal(mask_prob)}")
    for client_id in range(len(local_data)):
        raw_feature = local_data[client_id].x
        mask_file = osp.join(mask_root, f"mask_{client_id}.pt")
        if osp.exists(mask_file):
            mask = torch.load(mask_file)
        else:
            mask = torch.ones_like(raw_feature, dtype=torch.float32)
            for i in range(raw_feature.shape[0]):
                one_indices = raw_feature[i, :] != 0
                num_to_mask = int(mask_prob * one_indices.sum().item())
                indices_to_mask = torch.randperm(one_indices.sum())[:num_to_mask]
                mask[i, one_indices][indices_to_mask] = 0
            torch.save(mask, mask_file)        
        masked_feature = raw_feature * mask
        local_data[client_id].x = masked_feature
    return local_data