# 部分parameters同数据集高度相关
config = {
    "encoder": {
        "L": 2,
        "epochs": 50,
        "batch_size": 64,
        "hidden": 128,
        "out_channels": 7,
    },
    "cluster_batch_size": 32,
    "ae_pretrained_epochs": 2,
    "ae_finetune_epochs": 3,
    "dec_epochs": 5,
    "hide_portion": 0.5,
    "num_protos": 5,
    "num_preds": 5,
    "emb_shape": 128,
    "gen_hidden": 128,
    "feddep_epochs": 3,
    "pre_train_epochs": 2,
    "beta_d": 1.0,
    "beta_n": 1.0,
    "beta_c": 1.0
}
