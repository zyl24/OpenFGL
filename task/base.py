from torch.optim import Adam
    
class BaseTask:
    def __init__(self, args, client_id, data, data_dir, custom_model=None, custom_loss_fn=None):
        self.client_id = client_id
        self.data = data
        self.data_dir = data_dir
        self.args = args

        if custom_model is None:
            self.model = self.default_model
        else:
            self.model = custom_model

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        self.custom_loss_fn = custom_loss_fn
        
        self.load_train_val_test_split()
    
    
    def train(self):
        raise NotImplementedError
        
    def evaluate(self):
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
            
            
