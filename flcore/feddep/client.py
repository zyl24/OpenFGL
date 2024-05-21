import copy
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler

from flcore.base import BaseClient

from flcore.feddep.localdep import LocalDGen, FedDEP, Classifier_F
from flcore.feddep._utils import HideGraph, LocalRecLoss, FedRecLoss, GraphMender
from flcore.feddep.feddep_config import config


class FedDEPClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedDEPClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(Classifier_F(
            input_dim=(self.task.num_feats, config["emb_shape"]),
            hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes,
            num_layers=self.args.num_layers, dropout=self.args.dropout))

        self.hide_graph_model = HideGraph(hidden_portion=config["hide_portion"], num_preds=config["num_preds"], num_protos=config["num_protos"], device=device)
        
        
        self.data = self.task.splitted_data["data"]
        self.data.train_mask = self.task.splitted_data["train_mask"]
        self.data.val_mask = self.task.splitted_data["val_mask"]
        self.data.test_mask = self.task.splitted_data["test_mask"]
        
        
        self.hide_data, self.emb, self.x_missing = self.hide_graph_model(data=self.data)
        self.loss_fn_num = F.smooth_l1_loss
        self.loss_fn_rec = LocalRecLoss
        self.loss_fn_clf = F.cross_entropy

        self.send_message()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "embedding": self.emb,
            "x_missing": self.x_missing
        }

    def execute(self):
        # switch phase
        if self.message_pool["round"] == 0:
            self.phase = 0
        if self.message_pool["round"] == 1:
            self.phase = 1
            self.filled_data = GraphMender(
                model=self.feddep_model, impaired_data=self.hide_data,
                original_data=self.data, num_preds=config["num_preds"]).to(self.device)
            subgraph_sampler = NeighborSampler(
                self.data.edge_index, num_nodes=self.data.num_nodes,
                sizes=[-1], batch_size=4096, shuffle=False)
            self.fill_dataloader = {
                "data": self.filled_data,
                "train": NeighborSampler(
                    self.filled_data.edge_index,
                    num_nodes=self.filled_data.num_nodes,
                    node_idx=self.filled_data.train_idx,
                    sizes=[5, 5],
                    batch_size=64,
                    shuffle=True
                ),
                "val": subgraph_sampler,
                "test": subgraph_sampler
            }

        # execute
        if self.phase == 0:
            pre_train_model = LocalDGen(input_dim=self.task.num_feats, 
                emb_shape=config["emb_shape"],
                output_dim=self.task.num_global_classes, hid_dim=self.args.hid_dim,
                gen_dim=config["gen_hidden"], dropout=self.args.dropout,
                num_preds=config["num_preds"]).to(self.device)
            print(f"Client {self.client_id} pre-train start...")
            pre_train_model.train()
            pre_train_optim = self.task.default_optim(pre_train_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            for i in range(config["pre_train_epochs"]):
                pred_missing, pred_emb, nc_pred = pre_train_model(self.hide_data.to(self.device))
                mask_true_index = np.where(self.hide_data.train_mask.cpu().numpy() == True)[0]
                loss_num = self.loss_fn_num(
                    pred_missing[self.hide_data.train_mask],
                    self.hide_data.num_missing[self.hide_data.train_mask]
                )
                loss_rec = self.loss_fn_rec(
                    pred_embs=pred_emb[self.hide_data.train_mask],
                    true_embs=[self.hide_data.x_missing[node] for node in mask_true_index],
                    pred_missing=pred_missing[self.hide_data.train_mask],
                    true_missing=self.hide_data.num_missing[self.hide_data.train_mask],
                    num_preds=config["num_preds"]
                )
                loss_clf = self.loss_fn_clf(
                    nc_pred[self.hide_data.train_mask],
                    self.hide_data.y[self.hide_data.train_mask],
                )
                per_train_loss = config["beta_d"] * loss_num + config["beta_c"] * loss_clf
                per_train_loss += config["beta_n"] * loss_rec

                pre_train_optim.zero_grad()
                per_train_loss.backward()
                pre_train_optim.step()
                print(f"Client {self.client_id} local pre-train @Epoch {i}.")
            print(f"Client {self.client_id} pre-train finish!")

            self.feddep_model = FedDEP(pre_train_model).to(self.device)
            feddep_optim = self.task.default_optim(self.feddep_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            # self.task.model.phase += 1
            for i in range(config["feddep_epochs"]):
                dep_grad = dict()
                para_backup = copy.deepcopy(self.feddep_model.state_dict())
                for client_id in self.message_pool["sampled_clients"]:
                    if client_id != self.client_id:
                        # calculate gradients
                        emb, x_missing = (
                            self.message_pool[f"client_{client_id}"]["embedding"],
                            self.message_pool[f"client_{client_id}"]["x_missing"])
                        self.feddep_model.load_state_dict(para_backup)
                        self.feddep_model.train()
                        _, embedding = self.feddep_model.encoder_model(self.hide_data)
                        pred_missing = self.feddep_model.reg_model(embedding)
                        pred_embs = self.feddep_model.gen(embedding)
                        emb_len = pred_embs.shape[-1] // config["num_preds"]

                        choice = np.random.choice(len(x_missing), embedding.shape[0])
                        global_target_emb = []
                        for c_i in choice:
                            choice_i = np.random.choice(
                                len(x_missing[c_i]), config["num_preds"])
                            for ch_i in choice_i:
                                if torch.sum(x_missing[c_i][ch_i]) < 1e-15:
                                    global_target_emb.append(emb[c_i])
                                else:
                                    global_target_emb.append(
                                        x_missing[c_i][ch_i].detach().cpu().numpy())
                        global_target_emb = np.asarray(global_target_emb).reshape(
                            (embedding.shape[0], config["num_preds"], emb_len))

                        loss_emb = FedRecLoss(
                            pred_embs=pred_embs,
                            true_embs=global_target_emb,
                            pred_missing=pred_missing,
                            num_preds=config["num_preds"],
                        )
                        other_loss = (
                            1.0 / self.args.num_clients * config["beta_n"] * loss_emb
                        ).requires_grad_()
                        other_loss.backward()
                        # sum up all gradients from other clients
                        if not dep_grad:
                            for k, v in self.feddep_model.named_parameters():
                                dep_grad[k] = v.grad
                        else:
                            for k, v in self.feddep_model.named_parameters():
                                dep_grad[k] += v.grad
                # Rollback
                self.feddep_model.load_state_dict(para_backup)

                pred_missing, pred_emb, nc_pred = self.feddep_model.forward(self.hide_data)
                mask_true_index = np.where(self.hide_data.train_mask.cpu().numpy() == True)[0]
                loss_num = self.loss_fn_num(
                    pred_missing[self.hide_data.train_mask],
                    self.hide_data.num_missing[self.hide_data.train_mask]
                )
                loss_rec = self.loss_fn_rec(
                    pred_embs=pred_emb[self.hide_data.train_mask],
                    true_embs=[self.hide_data.x_missing[node] for node in mask_true_index],
                    pred_missing=pred_missing[self.hide_data.train_mask],
                    true_missing=self.hide_data.num_missing[self.hide_data.train_mask],
                    num_preds=config["num_preds"]
                )
                loss_clf = self.loss_fn_clf(
                    nc_pred[self.hide_data.train_mask],
                    self.hide_data.y[self.hide_data.train_mask],
                )
                feddep_loss = config["beta_d"] * loss_num + config["beta_c"] * loss_clf
                feddep_loss += config["beta_n"] * loss_rec
                feddep_loss = feddep_loss.float() / self.args.num_clients

                feddep_optim.zero_grad()
                feddep_loss.backward()
                feddep_optim.step()

                for k, v in self.feddep_model.named_parameters():
                    v.grad += dep_grad[k]
                feddep_optim.step()
        else:
            for (local_param, global_param) in zip(
                self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)
            for data_batch in self.fill_dataloader["train"]:
                batch_size, n_id, adjs = data_batch
                adjs = [adj.to(self.device) for adj in adjs]
                if "mend_emb" not in self.fill_dataloader["data"]:
                    mend_emb = torch.zeros(
                        (len(self.fill_dataloader["data"].x), self.task.model.emb_len)
                    ).to(self.device)
                else:
                    mend_emb = self.fill_dataloader["data"].mend_emb
                pred = self.task.model.forward(
                    (self.fill_dataloader["data"].x[n_id], mend_emb[n_id]), adjs=adjs)
                label = self.fill_dataloader["data"].y[n_id[:batch_size]].to(self.device)
                loss_clf = self.loss_fn_clf(pred, label)
                self.task.optim.zero_grad()
                loss_clf.backward()
                self.task.optim.step()
