import os.path as osp
import os
from decimal import Decimal
import torch
import numpy as np
import copy
from torch_geometric.loader import DataLoader
import random




def processing(args, splitted_data, processed_dir, client_id):
    if args.processing == "raw":
        processed_data = splitted_data
    elif args.processing == "random_feature_sparsity":
        processed_data = random_feature_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_feature_noise":
        from data.processing import random_feature_noise
        processed_data = random_feature_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, noise_std=args.processing_percentage)
    elif args.processing == "random_edge_sparsity":
        from data.processing import random_edge_sparsity
        processed_data = random_edge_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_edge_noise":
        from data.processing import random_edge_noise
        processed_data = random_edge_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, epsilon=args.processing_percentage, num_choice=2)
    elif args.processing == "random_label_sparsity":
        from data.processing import random_label_sparsity
        processed_data = random_label_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_label_noise":
        from data.processing import random_label_noise
        processed_data = random_label_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, noise_prob=args.processing_percentage)
    
    return processed_data
        
            
            
            
            
# Feature

def random_feature_sparsity(args, splitted_data: dict, processed_dir: str, client_id: int, mask_prob: float=0.1):
    mask_filename = osp.join(processed_dir, "mask", f"random_feature_mask_{mask_prob:.2f}_client_{client_id}.pt")
    
    if args.task == "node_cls":
        for name in ["data", "train_mask", "val_mask", "test_mask"]:
            assert name in splitted_data
            
        if osp.exists(mask_filename):
            mask = torch.load(mask_filename)
        else:
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            mask = torch.ones_like(splitted_data["data"].x, dtype=torch.float32)
            for i in range(splitted_data["data"].x.shape[0]):
                one_indices = splitted_data["data"].x[i, :] != 0
                num_to_mask = int(mask_prob * one_indices.sum().item())
                indices_to_mask = torch.randperm(one_indices.sum())[:num_to_mask]
                mask[i, one_indices][indices_to_mask] = 0
                torch.save(mask, mask_filename)        
        
        masked_splitted_data = copy.deepcopy(splitted_data)
        masked_splitted_data["data"].x *= mask
    elif args.task == "graph_cls":
        for name in ["data", "train_mask", "val_mask", "test_mask", "train_dataloader", "val_dataloader", "test_dataloader"]:
            assert name in splitted_data
        if osp.exists(mask_filename):
            mask = torch.load(mask_filename)
        else:
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            mask = torch.ones_like(splitted_data["data"].x, dtype=torch.float32)
            for i in range(splitted_data["data"].x.shape[0]):
                nonzero = (splitted_data["data"].x[i, :]).nonzero().squeeze().tolist()
                if type(nonzero) is not list:
                    nonzero = [nonzero]
                num_to_mask = int(mask_prob * len(nonzero))
                random.shuffle(nonzero)
                mask[i, nonzero[:num_to_mask]] = 0
            torch.save(mask, mask_filename)
        
        mask = mask.to(splitted_data["data"].data.x.device)
        masked_splitted_data = copy.deepcopy(splitted_data)
        masked_splitted_data["data"].data.x *= mask

        masked_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][splitted_data["train_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["val_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][splitted_data["val_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["test_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][splitted_data["test_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)

        
        # check
        for batch1, batch2 in zip(splitted_data["train_dataloader"], masked_splitted_data["train_dataloader"]):
            print(batch1.x)
            print(batch2.x)
            print("---"*30)
        
    return masked_splitted_data




def random_feature_noise(splitted_data: dict, processed_dir: str, client_id: int, noise_std: float=0.1):
    noise_filename = osp.join(processed_dir, "noise", f"random_feature_noise_{noise_std:.2f}_client_{client_id}.pt")
 
    if os.path.exists(noise_filename):
        noise = torch.load(noise_filename)
    else:
        os.makedirs(os.path.dirname(noise_filename), exist_ok=True)
        noise = torch.randn_like(splitted_data["data"].x) * noise_std
        torch.save(noise, noise_filename)
         
    noised_splitted_data = copy.deepcopy(splitted_data)
    noised_splitted_data["data"].x += noise
        
    return noised_splitted_data
 


# Label


def random_label_sparsity(splitted_data: dict, processed_dir: str, client_id: int, mask_prob=0.1):
    mask_filename = osp.join(processed_dir, "mask", f"random_label_mask_{mask_prob:.2f}_client_{client_id}.pt")

    if osp.exists(mask_filename):
        masked_train_mask = torch.load(mask_filename)
    else:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        masked_train_mask = torch.clone(splitted_data["train_mask"])
        indices = torch.nonzero(masked_train_mask, as_tuple=True)[0]
        num_samples = int(round(len(indices) * mask_prob))
        selected_indices = np.random.choice(indices.tolist(), num_samples, replace=False)
        masked_train_mask[selected_indices] = 0
        torch.save(masked_train_mask, mask_filename)
        
        
    masked_splitted_data = copy.deepcopy(splitted_data)
    masked_splitted_data["train_mask"] = masked_train_mask

    return masked_splitted_data


def random_label_noise(splitted_data: dict, processed_dir: str, client_id: int, noise_prob: float=0.1):
    noise_filename = osp.join(processed_dir, "noise", f"random_label_noise_{noise_prob:.2f}_client_{client_id}.pt")

    if osp.exists(noise_filename):
        noised_label = torch.load(noise_filename)
    else:
        os.makedirs(os.path.dirname(noise_filename), exist_ok=True)
        
        
        noised_label = torch.clone(splitted_data["data"].y)
        all_labels = [class_i for class_i in range(splitted_data["data"].num_global_classes)]

        train_mask = splitted_data["train_mask"]
        indices = torch.nonzero(train_mask, as_tuple=True)[0]
        num_samples = int(round(len(indices) * noise_prob))
        selected_indices = np.random.choice(indices.tolist(), num_samples, replace=False)
        for idx in selected_indices:
            real_label = splitted_data["data"].y[idx].item()
            new_label = real_label
            while(new_label == real_label):
                new_label = np.random.choice(all_labels)                
            noised_label[idx] = new_label
        
        torch.save(noised_label, noise_filename)
        
    noised_splitted_data = copy.deepcopy(splitted_data)
    noised_splitted_data["data"].y = noised_label
    
    return noised_splitted_data






 
 
# Topology 

def random_edge_sparsity(splitted_data: dict, processed_dir: str, client_id: int, mask_prob: float=0.1):
    masked_edge_index_filename = osp.join(processed_dir, "mask", f"random_edge_mask_{mask_prob:.2f}_client_{client_id}.pt")

    if osp.exists(masked_edge_index_filename):
        masked_edge_index = torch.load(masked_edge_index_filename)
    else:
        os.makedirs(os.path.dirname(masked_edge_index_filename), exist_ok=True)
    
        edge_list = splitted_data["data"].edge_index.T.tolist()
        num_nodes = splitted_data["data"].x.shape[0]
        masked_edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if ([i,j] in edge_list) or ([j,i] in edge_list):
                    # undirected graph
                    rnd = np.random.random()
                    if rnd >= mask_prob:
                        masked_edge_index.append((i, j))
                        masked_edge_index.append((j, i))

        masked_edge_index = torch.tensor(masked_edge_index).T
        torch.save(masked_edge_index, masked_edge_index_filename)  
    
    
    masked_splitted_data = copy.deepcopy(splitted_data)
    masked_splitted_data["data"].edge_index = masked_edge_index
    
    return masked_splitted_data

    
def random_edge_noise(splitted_data: dict, processed_dir: str, client_id: int, epsilon: float=0., num_choice: int=2):
    noised_edge_index_filename = osp.join(processed_dir, "noise", f"random_edge_noise_{epsilon:.2f}_client_{client_id}.pt")

    if osp.exists(noised_edge_index_filename):
        noised_edge_index = torch.load(noised_edge_index_filename)
    else:
        os.makedirs(os.path.dirname(noised_edge_index_filename), exist_ok=True)
        prob = np.e ** epsilon / (np.e ** epsilon + num_choice - 1)
        
        edge_list = splitted_data["data"].edge_index.T.tolist()
        num_nodes = splitted_data["data"].x.shape[0]
        noised_edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if ([i,j] in edge_list) or ([j,i] in edge_list):
                    # undirected graph
                    rnd = np.random.random()
                    if rnd <= prob:
                        noised_edge_index.append((i, j))
                        noised_edge_index.append((j, i))
                else:
                    rnd = np.random.random()
                    if rnd > prob:
                        noised_edge_index.append((i, j))
                        noised_edge_index.append((j, i))

        noised_edge_index = torch.tensor(noised_edge_index).T
        
        torch.save(noised_edge_index, noised_edge_index_filename)  
    
    
    masked_splitted_data = copy.deepcopy(splitted_data)
    masked_splitted_data["data"].edge_index = noised_edge_index
    






# def random_home_edge_injection(local_data, process_dir, ratio=0.):
#     assert not isinstance(ratio, list)

#     homo_random_injection_root = osp.join(process_dir, f"homo_injection_{ratio}")
#     if not osp.exists(homo_random_injection_root):
#         os.makedirs(homo_random_injection_root)
#     for client_id in range(len(local_data)):
#         local_labels = local_data[client_id].y
#         edge_file = osp.join(homo_random_injection_root, f"edge_list_{client_id}.pt")
#         if osp.exists(edge_file):
#             new_edge_list = torch.load(edge_file)
#         else:
#             edge_list = local_data[client_id].edge_index.T.tolist()
#             num_nodes = local_data[client_id].num_nodes
#             # homo_cnt = 0
#             # homo_added = 0
#             # print(num_nodes)
#             # print(len(edge_list))
#             new_edge_list = edge_list
#             for i in range(num_nodes):
#                 for j in range(i+1, num_nodes):
#                     if local_labels[i] == local_labels[j]:
#                         if ([i,j] not in edge_list) and ([j,i] not in edge_list):
#                             # undirected graph
#                             # homo_cnt += 1
#                             rnd = np.random.random()
#                             if rnd < ratio:
#                                 homo_added += 1
#                                 new_edge_list.append((i, j))
#                                 new_edge_list.append((j, i))

#             new_edge_list = torch.tensor(new_edge_list).T
#             torch.save(new_edge_list, edge_file)  
            
#         local_data[client_id].edge_index = new_edge_list
#         # print(homo_cnt, homo_added)
#         # print(local_data[client_id].edge_index.shape)
#     return local_data

# def random_hete_edge_injection(local_data, process_dir, ratio=0.):
#     assert not isinstance(ratio, list)

#     hete_random_injection_root = osp.join(process_dir, f"hete_injection_{ratio}")
#     if not osp.exists(hete_random_injection_root):
#         os.makedirs(hete_random_injection_root)
#     for client_id in range(len(local_data)):
#         local_labels = local_data[client_id].y
#         edge_file = osp.join(hete_random_injection_root, f"edge_list_{client_id}.pt")
#         if osp.exists(edge_file):
#             edge_list = torch.load(edge_file)
#         else:
#             edge_list = local_data[client_id].edge_index.T.tolist()
#             num_nodes = local_data[client_id].num_nodes
#             new_edge_list = edge_list
#             for i in range(num_nodes):
#                 for j in range(i+1, num_nodes):
#                     if local_labels[i] != local_labels[j]:
#                         if ([i,j] not in edge_list) and ([j,i] not in edge_list):
#                             # undirected graph
#                             rnd = np.random.random()
#                             if rnd < ratio:
#                                 new_edge_list.append((i, j))
#                                 new_edge_list.append((j, i))

#             new_edge_list = torch.tensor(new_edge_list).T
#             torch.save(new_edge_list, edge_file)  
            
#         local_data[client_id].edge_index = new_edge_list
#     return local_data


