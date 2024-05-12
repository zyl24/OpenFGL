import os
import copy

class Logger:
    
    def __init__(self, args, message_pool, task_path):
        self.args = args
        self.message_pool = message_pool
        self.debug = self.args.debug
        self.task_path = task_path
        self.log_list = []
        
    @property
    def log_path(self):
        return os.path.join(self.task_path, "debug", self.args.fl_algorithm)
    
    
    def get_round_logs(self, evaluation_result):
        # 1. save task-oriented metric in each round
        self.log_list.append(copy.deepcopy(evaluation_result))
    
        # 2. auto compute bandwidth
        total_bytes = 0

        # # # 遍历字典，找到所有的 tensor 并计算总字节数
        # for key, value in tensor_dict.items():
        #     if torch.is_tensor(value):
        #         # 计算当前 tensor 的字节数
        #         bytes_of_tensor = value.element_size() * value.nelement()
        #         total_bytes += bytes_of_tensor
        #         print(f"Tensor '{key}' has {bytes_of_tensor} bytes.")