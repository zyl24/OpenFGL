from torch.optim import Adam
    
class BaseTask:
    def __init__(self, args, client_id, data, data_dir, device):
        self.client_id = client_id
        self.data = data.to(device)
        self.data_dir = data_dir
        self.args = args
        self.device = device
        self.model = self.default_model
        self.model = self.model.to(device)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.load_train_val_test_split()

        self.override_evaluate = None
    
    def train(self):
        raise NotImplementedError
        
    def evaluate(self):
        raise NotImplementedError
    
    
    @property
    def num_samples(self):
        raise NotImplementedError
    
    @property
    def default_model(self):
        raise NotImplementedError
    
    @property
    def default_optim(self):
        raise NotImplementedError
    
    @property
    def default_loss_fn(self):
        raise NotImplementedError
    
    @property
    def train_val_test_path(self):
        raise NotImplementedError
    
    @property
    def default_train_val_test_split(self):
        raise NotImplementedError    

    def load_train_val_test_split(self):
        raise NotImplementedError
    
    def load_custom_model(self, custom_model):
        self.model = custom_model.to(self.device)
        self.optim = self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

            
            
