import torch
import torch.nn as nn
from utils.basic_utils import load_task


class BaseClient:
    def __init__(self, args, client_id, data, data_dir, message_pool, device, custom_model=None):
        self.args = args
        self.client_id = client_id
        self.data = data
        self.data_dir = data_dir
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, client_id, data, data_dir, device, custom_model)        
    
    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError


    

class BaseServer:
    def __init__(self, args, global_data, data_dir, message_pool, device, custom_model=None):
        self.args = args
        self.message_pool = message_pool
        self.global_data = global_data
        self.data_dir = data_dir
        self.device = device
        self.task = load_task(args, None, global_data, data_dir, device, custom_model)

   
    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError