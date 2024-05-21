import torch
from flcore.base import BaseServer

from flcore.feddep.localdep import Classifier_F
from flcore.feddep.feddep_config import config


class FedDEPEServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedDEPEServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(Classifier_F(
            input_dim=(self.task.num_feats, config["emb_shape"]),
            hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes,
            num_layers=self.args.num_layers, dropout=self.args.dropout))
        self.task.override_evaluate = self.get_phase_1_override_evaluate()

    def execute(self):
        if self.message_pool["round"] == 0:
            pass
        else:
            with torch.no_grad():
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = 1 / len(self.message_pool["sampled_clients"])
                    for local_param, global_param in zip(
                        self.message_pool[f"client_{client_id}"]["weight"],
                        self.task.model.parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param

    def send_message(self):
        self.message_pool["server"] = {"weight": list(self.task.model.parameters())}

    def get_phase_1_override_evaluate(self):
        from utils.metrics import compute_supervised_metrics

        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data

            eval_output = {}
            self.task.model.eval()
            with torch.no_grad():
                logits = self.task.model.forward(splitted_data)

            metric_train = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["train_mask"]],
                labels=splitted_data.y[splitted_data["train_mask"]],
                suffix="train"
            )
            metric_val = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["val_mask"]],
                labels=splitted_data.y[splitted_data["val_mask"]],
                suffix="val"
            )
            metric_test = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["test_mask"]],
                labels=splitted_data.y[splitted_data["test_mask"]],
                suffix="test"
            )
            eval_output = {**metric_train, **metric_val, **metric_test}

            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = "[server]"
            if not mute:
                print(prefix + info)
            return eval_output

        return override_evaluate
