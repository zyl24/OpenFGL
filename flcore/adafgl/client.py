import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.adafgl.adafgl_models import AdaFGLModel
from flcore.adafgl.adafgl_config import config
from flcore.adafgl._utils import adj_initialize
from torch.optim import Adam
import torch.nn.functional as F


class AdaFGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(AdaFGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        
        self.phase = 0

        
        
    def execute(self):
        # switch phase
        if self.message_pool["round"] == config["num_rounds_vanilla"]:
            self.phase = 1
            self.adafgl_model = AdaFGLModel(prop_steps=config["prop_steps"], feat_dim=self.task.num_feats, hidden_dim=self.args.hid_dim, output_dim=self.task.num_global_classes, train_mask=self.task.train_mask, val_mask=self.task.val_mask, test_mask=self.task.test_mask, r=config["r"])
            self.task.data = adj_initialize(self.task.data)
            
            # last time download global model
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)
            
            # initialize adafgl model
            eval_output = self.task.evaluate()
            node_logits = eval_output["logits"]
            soft_label = nn.Softmax(dim=1)(node_logits)
            self.adafgl_model.non_para_lp(subgraph=self.task.data, soft_label=soft_label, x=self.task.data.x, device=self.device)
            self.adafgl_model.preprocess(adj=self.task.data.adj)
            self.adafgl_model = self.adafgl_model.to(self.device)
            
            # create optimizer
            self.adafgl_optimizer = Adam(self.adafgl_model.parameters(), lr=config["adafgl_lr"], weight_decay=config["adafgl_weight_decay"])
        
            # override evaluation
            self.task.override_evaluate = self.get_adafgl_override_evaluate()
            
            
        # execute
        if self.phase == 0:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)
            self.task.train()
        else:
            self.adafgl_postprocess()

        

    def send_message(self):
        if self.phase == 0:
            self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        else:
            self.message_pool[f"client_{self.client_id}"] = {}
        
    def adafgl_postprocess(self, loss_ce_fn=nn.CrossEntropyLoss()):
        self.adafgl_model.train()
        self.adafgl_optimizer.zero_grad()

        # homo forward
        local_smooth_logits, global_logits = self.adafgl_model.homo_forward(self.device)
        loss_train1 = loss_ce_fn(local_smooth_logits[self.task.train_mask], self.task.data.y[self.task.train_mask])
        loss_train2 = nn.MSELoss()(local_smooth_logits, global_logits)
        loss_train_homo = loss_train1 + loss_train2
            
        # hete forward
        local_ori_logits, local_smooth_logits, local_message_propagation = self.adafgl_model.hete_forward(self.device)
        loss_train1 = loss_ce_fn(local_ori_logits[self.task.train_mask], self.task.data.y[self.task.train_mask])
        loss_train2 = loss_ce_fn(local_smooth_logits[self.task.train_mask], self.task.data.y[self.task.train_mask])
        loss_train3 = loss_ce_fn(local_message_propagation[self.task.train_mask], self.task.data.y[self.task.train_mask])
        loss_train_hete = loss_train1 + loss_train2 + loss_train3
        
        # final loss
        loss_final = loss_train_homo + loss_train_hete

        loss_final.backward()
        self.adafgl_optimizer.step()


    def get_adafgl_override_evaluate(self):
        from utils.metrics import compute_supervised_metrics
        def override_evaluate(splitted_data=None, mute=False):
            assert splitted_data is None, "AdaFGL doesn't support global data evaluation."
            splitted_data = self.task.splitted_data
            
            eval_output = {}    
            self.adafgl_model.eval()    
            
            
            # homo eval
            with torch.no_grad():
                local_smooth_logits, global_logits = self.adafgl_model.homo_forward(self.device)
                output_homo = (F.softmax(local_smooth_logits.data, 1) + F.softmax(global_logits.data, 1)) / 2


                # hete eval
                local_ori_logits, local_smooth_logits, local_message_propagation = self.adafgl_model.hete_forward(self.device)
                output_hete = (F.softmax(local_ori_logits.data, 1) + F.softmax(local_smooth_logits.data, 1) + F.softmax(
                    local_message_propagation.data, 1)) / 3
                
                # merge
                homo_weight = self.adafgl_model.reliability_acc 
                embedding = None
                logits = homo_weight * output_homo + (1- homo_weight) * output_hete
                
                
                loss_train = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
                loss_val = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
                loss_test = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            return eval_output
        
        return override_evaluate

