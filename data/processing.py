import os.path as osp
import os
from decimal import Decimal
import torch
import numpy as np

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

def random_feature_mask(local_data, process_dir, mask_prob=0.1):
    mask_root = osp.join(process_dir, f"mask_{mask_prob}")
    if not osp.exists(mask_root):
        os.makedirs(mask_root)
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

def link_random_response(local_data, process_dir, epsilon=0., num_choice=2):
    assert not isinstance(epsilon, list)

    link_random_response_root = osp.join(process_dir, f"link_random_response_{epsilon}")
    if not osp.exists(link_random_response_root):
        os.makedirs(link_random_response_root)
    prob = np.e ** epsilon / (np.e ** epsilon + num_choice - 1)
    print(prob)
    for client_id in range(len(local_data)):
        edge_file = osp.join(link_random_response_root, f"edge_list_{client_id}.pt")
        if osp.exists(edge_file):
            new_edge_list = torch.load(edge_file)
        else:
            edge_list = local_data[client_id].edge_index.T.tolist()
            num_nodes = local_data[client_id].num_nodes
            new_edge_list = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if ([i,j] in edge_list) or ([j,i] in edge_list):
                        # undirected graph
                        rnd = np.random.random()
                        if rnd <= prob:
                            new_edge_list.append((i, j))
                            new_edge_list.append((j, i))
                    else:
                        rnd = np.random.random()
                        if rnd > prob:
                            new_edge_list.append((i, j))
                            new_edge_list.append((j, i))

            new_edge_list = torch.tensor(new_edge_list).T
            torch.save(new_edge_list, edge_file)  
            
        local_data[client_id].edge_index = new_edge_list
    return local_data

def homo_random_injection(local_data, process_dir, ratio=0.):
    assert not isinstance(ratio, list)

    homo_random_injection_root = osp.join(process_dir, f"homo_injection_{ratio}")
    if not osp.exists(homo_random_injection_root):
        os.makedirs(homo_random_injection_root)
    for client_id in range(len(local_data)):
        local_labels = local_data[client_id].y
        edge_file = osp.join(homo_random_injection_root, f"edge_list_{client_id}.pt")
        if osp.exists(edge_file):
            new_edge_list = torch.load(edge_file)
        else:
            edge_list = local_data[client_id].edge_index.T.tolist()
            num_nodes = local_data[client_id].num_nodes
            # homo_cnt = 0
            # homo_added = 0
            # print(num_nodes)
            # print(len(edge_list))
            new_edge_list = edge_list
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if local_labels[i] == local_labels[j]:
                        if ([i,j] not in edge_list) and ([j,i] not in edge_list):
                            # undirected graph
                            # homo_cnt += 1
                            rnd = np.random.random()
                            if rnd < ratio:
                                homo_added += 1
                                new_edge_list.append((i, j))
                                new_edge_list.append((j, i))

            new_edge_list = torch.tensor(new_edge_list).T
            torch.save(new_edge_list, edge_file)  
            
        local_data[client_id].edge_index = new_edge_list
        # print(homo_cnt, homo_added)
        # print(local_data[client_id].edge_index.shape)
    return local_data

def hete_random_injection(local_data, process_dir, ratio=0.):
    assert not isinstance(ratio, list)

    hete_random_injection_root = osp.join(process_dir, f"hete_injection_{ratio}")
    if not osp.exists(hete_random_injection_root):
        os.makedirs(hete_random_injection_root)
    for client_id in range(len(local_data)):
        local_labels = local_data[client_id].y
        edge_file = osp.join(hete_random_injection_root, f"edge_list_{client_id}.pt")
        if osp.exists(edge_file):
            edge_list = torch.load(edge_file)
        else:
            edge_list = local_data[client_id].edge_index.T.tolist()
            num_nodes = local_data[client_id].num_nodes
            new_edge_list = edge_list
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if local_labels[i] != local_labels[j]:
                        if ([i,j] not in edge_list) and ([j,i] not in edge_list):
                            # undirected graph
                            rnd = np.random.random()
                            if rnd < ratio:
                                new_edge_list.append((i, j))
                                new_edge_list.append((j, i))

            new_edge_list = torch.tensor(new_edge_list).T
            torch.save(new_edge_list, edge_file)  
            
        local_data[client_id].edge_index = new_edge_list
    return local_data