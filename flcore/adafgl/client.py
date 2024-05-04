import torch
import torch.nn as nn
from flcore.base import BaseClient
import time

class AdaFGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(AdaFGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.train()
        
        if self.message_pool["round"] == self.args.num_rounds - 1:
            self.adafgl_postprocess()
        

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        
        
    def adafgl_postprocess(self):
        main_normalize_train = 1

        print("Start AdaFGL personalized local training...")
        global_normalize_record = {"acc_val_mean": 0, "acc_val_std": 0, "acc_test_mean": 0, "acc_test_std": 0}

        t_total = time.time()
    # for i in range(len(datasets.subgraphs)):
    #     subgraph = datasets.subgraphs[i]
    #     subgraph.y = subgraph.y.to(device)
    #     local_normalize_record = {"acc_val": [], "acc_test": []}
    #     for _ in range(main_normalize_train):
    #         gmodel = torch.load(osp.join("./model_weights",
    #                                      "{}_Client{}_{}_{}.pt".format(datasets.name, datasets.num_clients,
    #                                                                    datasets.sampling, gmodel_name)))
    #         gmodel.preprocess(subgraph.adj, subgraph.x)
    #         gmodel = gmodel.to(device)
    #         nodes_embedding = gmodel.model_forward(range(subgraph.num_nodes), device).detach().cpu()
    #         nodes_embedding = nn.Softmax(dim=1)(nodes_embedding)
    #         acc_val = accuracy(nodes_embedding[subgraph.val_idx], subgraph.y[subgraph.val_idx])
    #         acc_test = accuracy(nodes_embedding[subgraph.test_idx], subgraph.y[subgraph.test_idx])
    #         model = MyModel(prop_steps=3,
    #                         feat_dim=datasets.input_dim,
    #                         hidden_dim=args.hidden_dim,
    #                         output_dim=datasets.output_dim,
    #                         threshold=args.threshold)

    #         model.non_para_lp(subgraph=subgraph, nodes_embedding=nodes_embedding, x=subgraph.x, device=device)
    #         model.preprocess(adj=subgraph.adj)
    #         model = model.to(device)
    #         loss_ce_fn = nn.CrossEntropyLoss()
    #         loss_mse_fn = nn.MSELoss()
    #         optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #         epochs = args.epochs
    #         best_val = 0.
    #         best_test = 0.
    #         homo_best_val_global = 0.
    #         homo_best_test_global = 0.
    #         homo_best_val_local = 0.
    #         homo_best_test_local = 0.
    #         hete_best_val_smooth = 0.
    #         hete_best_test_smooth = 0.
    #         hete_best_val_local = 0.
    #         hete_best_test_local = 0.
    #         hete_best_val_prop = 0.
    #         hete_best_test_prop = 0.
    #         best_epoch = 0
    #         for epoch in range(epochs):
    #             t = time.time()
    #             model.train()
    #             optimizer.zero_grad()
    #             if model.homo:
    #                 local_smooth_emb, global_emb = model.homo_forward(device)
    #                 loss_train1 = loss_ce_fn(local_smooth_emb[subgraph.train_idx], subgraph.y[subgraph.train_idx])
    #                 loss_train2 = nn.MSELoss()(local_smooth_emb, global_emb)
    #                 loss_train = loss_train1 + loss_train2
    #                 loss_train.backward()
    #                 optimizer.step()
    #                 model.eval()
    #                 local_smooth_emb, global_emb = model.homo_forward(device)
    #                 output = (F.softmax(local_smooth_emb.data, 1) + F.softmax(global_emb.data, 1)) / 2
    #                 acc_val = accuracy(output[subgraph.val_idx], subgraph.y[subgraph.val_idx])
    #                 acc_test = accuracy(output[subgraph.test_idx], subgraph.y[subgraph.test_idx])
    #                 homo_acc_val_local = accuracy(local_smooth_emb[subgraph.val_idx], subgraph.y[subgraph.val_idx])
    #                 homo_acc_test_local = accuracy(local_smooth_emb[subgraph.test_idx], subgraph.y[subgraph.test_idx])
    #                 homo_acc_val_global = accuracy(global_emb[subgraph.val_idx], subgraph.y[subgraph.val_idx])
    #                 homo_acc_test_global = accuracy(global_emb[subgraph.test_idx], subgraph.y[subgraph.test_idx])

    #             else:
    #                 local_ori_emb, local_smooth_emb, local_message_propagation = model.hete_forward(device)
    #                 loss_train1 = loss_ce_fn(local_ori_emb[subgraph.train_idx], subgraph.y[subgraph.train_idx])
    #                 loss_train2 = loss_ce_fn(local_smooth_emb[subgraph.train_idx], subgraph.y[subgraph.train_idx])
    #                 loss_train3 = loss_ce_fn(local_message_propagation[subgraph.train_idx],
    #                                          subgraph.y[subgraph.train_idx])
    #                 loss_train = loss_train1 + loss_train2 + loss_train3
    #                 loss_train.backward()
    #                 optimizer.step()
    #                 model.eval()
    #                 local_ori_emb, local_smooth_emb, local_message_propagation = model.hete_forward(device)
    #                 output = (F.softmax(local_ori_emb.data, 1) + F.softmax(local_smooth_emb.data, 1) + F.softmax(
    #                     local_message_propagation.data, 1)) / 3
    #                 acc_val = accuracy(output[subgraph.val_idx], subgraph.y[subgraph.val_idx])
    #                 acc_test = accuracy(output[subgraph.test_idx], subgraph.y[subgraph.test_idx])
    #                 hete_acc_val_local = accuracy(F.softmax(local_ori_emb[subgraph.val_idx], 1),
    #                                               subgraph.y[subgraph.val_idx])
    #                 hete_acc_test_local = accuracy(F.softmax(local_ori_emb[subgraph.test_idx], 1),
    #                                                subgraph.y[subgraph.test_idx])
    #                 hete_acc_val_smooth = accuracy(F.softmax(local_smooth_emb[subgraph.val_idx], 1),
    #                                                subgraph.y[subgraph.val_idx])
    #                 hete_acc_test_smooth = accuracy(F.softmax(local_smooth_emb[subgraph.test_idx], 1),
    #                                                 subgraph.y[subgraph.test_idx])
    #                 hete_acc_val_prop = accuracy(F.softmax(local_message_propagation[subgraph.val_idx], 1),
    #                                              subgraph.y[subgraph.val_idx])
    #                 hete_acc_test_prop = accuracy(F.softmax(local_message_propagation[subgraph.test_idx], 1),
    #                                               subgraph.y[subgraph.test_idx])
    #             if acc_val > best_val:
    #                 best_epoch = epoch + 1
    #                 best_val = acc_val
    #                 best_test = acc_test
    #             if model.homo:
    #                 if homo_acc_val_global > homo_best_val_global:
    #                     homo_best_val_global = homo_acc_val_global
    #                     homo_best_test_global = homo_acc_test_global
    #                 if homo_acc_val_local > homo_best_val_local:
    #                     homo_best_val_local = homo_acc_val_local
    #                     homo_best_test_local = homo_acc_test_local
    #             else:
    #                 if hete_acc_val_local > hete_best_val_local:
    #                     hete_best_val_local = hete_acc_val_local
    #                     hete_best_test_local = hete_acc_test_local
    #                 if hete_acc_val_smooth > hete_best_val_smooth:
    #                     hete_best_val_smooth = hete_acc_val_smooth
    #                     hete_best_test_smooth = hete_acc_test_smooth
    #                 if hete_acc_val_prop > hete_best_val_prop:
    #                     hete_best_val_prop = hete_acc_val_prop
    #                     hete_best_test_prop = hete_acc_test_prop
    #         local_normalize_record["acc_val"].append(best_val)
    #         local_normalize_record["acc_test"].append(best_test)

    #     global_normalize_record["acc_val_mean"] += np.mean(
    #         local_normalize_record["acc_val"]) * subgraph.num_nodes / datasets.global_data.num_nodes
    #     global_normalize_record["acc_val_std"] += np.std(local_normalize_record["acc_val"],
    #                                                      ddof=1) * subgraph.num_nodes / datasets.global_data.num_nodes
    #     global_normalize_record["acc_test_mean"] += np.mean(
    #         local_normalize_record["acc_test"]) * subgraph.num_nodes / datasets.global_data.num_nodes
    #     global_normalize_record["acc_test_std"] += np.std(local_normalize_record["acc_test"],
    #                                                       ddof=1) * subgraph.num_nodes / datasets.global_data.num_nodes

    # print("| ★  Normalize Train Completed")
    # print("| Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(args.normalize_train, time.time() - t_total))
    # print("| Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(
    #     round(global_normalize_record["acc_val_mean"], 4), round(global_normalize_record["acc_val_std"], 4),
    #     round(global_normalize_record["acc_test_mean"], 4), round(global_normalize_record["acc_test_std"], 4)))
    # return round(global_normalize_record["acc_test_mean"], 4)