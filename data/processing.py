import os.path as osp
import os
from decimal import Decimal
import torch
import numpy as np


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

def random_feature_noise(local_data, process_dir, noise_std=0.1):
    if isinstance(local_data, list):
        print(f"add Gaussian noise to features of all local data")
        noise_root = osp.join(process_dir, f"noise_{noise_std}")
        if not osp.exists(noise_root):
            os.makedirs(noise_root)
        for client_id in range(len(local_data)):
            raw_feature = local_data[client_id].x
            noise_file = osp.join(noise_root, f"noise_{client_id}.pt")
            if osp.exists(noise_file):
                noise = torch.load(noise_file)
            else:
                noise = torch.randn_like(raw_feature) * noise_std
                torch.save(noise, noise_file)
            noisy_feature = raw_feature + noise
            local_data[client_id].x = noisy_feature
        return local_data
    elif isinstance(local_data, dict):
        # NOTE deepcopy local_data
        print(f"add Gaussian noise to features of splitted data, including trainset and testset")
        noise_root = osp.join(process_dir, f"splitted_data_noise_{noise_std}")
        if not osp.exists(noise_root):
            os.makedirs(noise_root)
        if osp.exists(noise_root):
            noisy_data = torch.load(noise_root)
            local_data["data"].x = noisy_data
        else:
            raw_feature = local_data["data"].x
            noise = torch.randn_like(raw_feature) * noise_std
            noisy_feature = raw_feature + noise
            local_data["data"].x = noisy_feature
            # TODO save the results into file
        return local_data
        

def edge_random_mask(local_data, process_dir, mask_prob=0.1, num_choice=2):
    edge_random_mask_root = osp.join(process_dir, f"edge_mask_{mask_prob}")
    if not osp.exists(edge_random_mask_root):
        os.makedirs(edge_random_mask_root)
    for client_id in range(len(local_data)):
        edge_file = osp.join(edge_random_mask_root, f"edge_list_{client_id}.pt")
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
                        if rnd >= mask_prob:
                            new_edge_list.append((i, j))
                            new_edge_list.append((j, i))

            new_edge_list = torch.tensor(new_edge_list).T
            torch.save(new_edge_list, edge_file)  
            
        local_data[client_id].edge_index = new_edge_list
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

def label_noise(splitted_data, process_dir, percentage=0.1):
    # NOTE deepcopy splitted_data

    label_noise_root = osp.join(process_dir, f"splitted_data_label_noise_{percentage}")
    # if not osp.exists(label_noise_root):
    #     os.makedirs(label_noise_root)

    if osp.exists(label_noise_root):
        pass
        # noisy_label = torch.load(label_noise_root)
        # splitted_data["data"].y = noisy_label
    else:
        label_pool = splitted_data["data"].y.unique().tolist()
        assert len(label_pool) > 1

        train_mask = splitted_data["train_mask"]
        indices = torch.nonzero(train_mask, as_tuple=True)[0]
        num_samples = int(round(len(indices) * percentage))
        selected_indices = np.random.choice(indices.tolist(), num_samples, replace=False)
        for idx in selected_indices:
            raw_label = splitted_data["data"].y[idx].item()
            new_label = raw_label
            while(new_label == raw_label):
                new_label = np.random.choice(label_pool)
            splitted_data["data"].y[idx] = new_label
        
        # TODO save the results into file

    return splitted_data

def label_sparsity(splitted_data, process_dir, percentage=0.1):
    # NOTE deepcopy splitted_data

    label_sparsity_root = osp.join(process_dir, f"splitted_data_label_sparsity_{percentage}")
    # if not osp.exists(label_sparsity_root):
    #     os.makedirs(label_sparsity_root)
    if osp.exists(label_sparsity_root):
        # train_mask = torch.load(label_sparsity_root)
        # splitted_data["train_mask"] = train_mask
        pass
    else:
        train_mask = splitted_data["train_mask"]
        indices = torch.nonzero(train_mask, as_tuple=True)[0]
        # print(indices, indices.shape)
        num_samples = int(round(len(indices) * percentage))
        selected_indices = np.random.choice(indices.tolist(), num_samples, replace=False)
        train_mask[selected_indices] = False
        # indices = torch.nonzero(train_mask, as_tuple=True)[0]
        # print(indices, indices.shape)
        splitted_data["train_mask"] = train_mask
        # TODO save the results into file

    return splitted_data
