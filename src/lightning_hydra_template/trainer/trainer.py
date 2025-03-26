import torch


class Trainer():

    def __init__(
            self,
            model,
            optimizer_forward,
            optimizer_backward,
            train_config
    ):
        
        self.model = model

        self.optimizer_forward = optimizer_forward
        self.optimizer_backward = optimizer_backward

        self._init_config(train_config)

    def _init_config(
            self,
            train_config
    ):
        pass
    
    def train(
            self,
            train_loader_x0,
            train_loader_x1,
    ):
        
        for imf_iteration in range(self.n_imf):

            for batch_x0, batch_x1 in zip(train_loader_x0,train_loader_x1): #ligne 4

                self.train_step(batch_x0,batch_x1)
            
            # ligne 5-8
            new_data = self.model.generate_dataset()
        
    
    def train_step(
            self,
            batch
    )


