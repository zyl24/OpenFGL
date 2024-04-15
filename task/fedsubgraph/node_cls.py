import torch
import torch.nn as nn
from utils.basic_utils import extract_floats, idx_to_mask_tensor

def local_subgraph_train_val_test_split(local_subgraph, split, num_classes=None):
    num_nodes = local_subgraph.x.shape[0]
    train_, val_, test_ = extract_floats(split)
    
    local_subgraph.train_mask = idx_to_mask_tensor([], num_nodes)
    local_subgraph.val_mask = idx_to_mask_tensor([], num_nodes)
    local_subgraph.test_mask = idx_to_mask_tensor([], num_nodes)
    for class_i in range(num_classes):
        class_i_node_mask = local_subgraph.y == class_i
        num_class_i_nodes = class_i_node_mask.sum()
        local_subgraph.train_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_nodes)], num_nodes)
        local_subgraph.val_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
        local_subgraph.test_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_nodes): ], num_nodes)
    return local_subgraph



def accuracy(logits, label):
    pred = logits.max(1)[1]
    correct = (pred == label).sum()
    total = logits.shape[0]
    return correct / total * 100

    
    
class NodeClsTask():
    def __init__(self):
        pass
    
    @property
    def basic_loss_fn(self):
        return nn.CrossEntropyLoss()
    
    
    