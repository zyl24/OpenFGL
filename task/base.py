from torch.optim import Adam
    
class BaseTask:
    def __init__(self, args, client_id, data, data_dir, device, custom_model=None):
        self.client_id = client_id
        self.data = data.to(device)
        self.data_dir = data_dir
        self.args = args
        self.device = device
        

        if custom_model is None:
            self.model = self.default_model.to(device)
        else:
            self.model = custom_model.to(device)
            
        self.custom_loss_fn = None

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        self.load_train_val_test_split()
    
    
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
            
            
