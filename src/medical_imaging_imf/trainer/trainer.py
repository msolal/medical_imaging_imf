import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader, TensorDataset, StackDataset

from medical_imaging_imf.models.schrodinger_bridge import SchrodBridgeIMF

from abc import abstractmethod
from typing import Optional

from tqdm import tqdm

class Trainer():

    def __init__(
        self,
        model: SchrodBridgeIMF,
        optimizer_forward,
        optimizer_backward,
        train_config,
        ema: Optional[torch.nn.Module] = None,
    ):

        self._init_config(train_config)

        self.model = model.to(self.device)

        self.optimizer_forward = optimizer_forward
        self.optimizer_backward = optimizer_backward


    def _init_config(
            self,
            train_config
    ):
        self.n_imf = train_config.n_imf
        self.n_refresh = train_config.n_refresh
        self.n_epoch = train_config.n_epoch
                
        self.device = train_config.device
        self.n_device = train_config.n_device
        
        self.checkpoint_every_n_imf = train_config.checkpoint_every_n_imf
        self.check_val_every_n_refresh = train_config.check_val_every_n_refresh
        
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
    ):
        pass


class DSBMJointTrainer(Trainer):

    def __init__(
        self,
        model,
        optimizer_forward,
        optimizer_backward,
        train_config,
        ema,
    ):

        super().__init__(
            model,
            optimizer_forward,
            optimizer_backward,
            train_config,
            ema,
        )
        
        self.lambda_ = 1

    def train(
        self,
        train_loader: DataLoader,
    ):

        for imf_iteration in tqdm(range(self.n_imf), desc="IMF Iteration", leave=False):

            self.imf_iteration = imf_iteration

            for refresh_iteration in tqdm(range(self.n_refresh), desc="Refresh Iteration", leave=False):

                # NOTE : could load old model weights here
                new_train_loader = self._generate_new_train_loader(train_loader)

                for epoch in tqdm(range(self.n_epoch), desc="Epoch", leave=False):

                    for x0, x1 in tqdm(new_train_loader, desc="Batch", leave=False):
                        
                        x0 = x0.squeeze(-1).to(self.device)
                        x1 = x1.squeeze(-1).to(self.device)

                        self.optimizer_forward.zero_grad()
                        self.optimizer_backward.zero_grad()

                        self.train_step(x0, x1)


    def train_step(
        self,
        x0: TensorType,
        x1: TensorType,
    ):

        # TODO cleaner shape handling for t
        t = torch.rand((x0.shape[0], 1, 1, 1), device=self.device)
        z = torch.randn_like(x0)

        loss = (
            self.model.loss_forward(x0, x1, z, t)
            + self.model.loss_backward(x0, x1, z, t)
            + self.lambda_ * self.model.loss_consistency(x0, x1, z, t)
        )

        # loss.backward()

        # self.optimizer_forward.step()
        # self.optimizer_backward.step()


    def _generate_new_train_loader(
        self,
        train_loader: DataLoader,

    ):
        
        self.img_size = train_loader.dataset[0][0].shape[1:-1]
        
        new_tensor_x0 = torch.zeros((len(train_loader.dataset), 1, *self.img_size))
        new_tensor_x1 = torch.zeros((len(train_loader.dataset), 1, *self.img_size)) 

        index = 0
        # TODO change back to != 0
        if self.imf_iteration == 0:
            for x0, x1 in train_loader:
                
                x0 = x0.squeeze(-1).to(self.device)
                x1 = x1.squeeze(-1).to(self.device)
                x0, x1 = self.model.mixture(x0, x1)
                
                # new_tensor has shape (total_n_slices, 2, channel, height, width)
                new_tensor_x0[index: index + x0.shape[0], ...] = x0
                new_tensor_x1[index: index + x0.shape[0], ...] = x1

                index += x0.shape[0]
            
            new_tensor_dataset = TensorDataset(new_tensor_x0, new_tensor_x1)
            new_train_loader = DataLoader(new_tensor_dataset, batch_size = train_loader.batch_size)
        else:
            new_train_loader = train_loader

        return new_train_loader

class DSBMTrainer(Trainer):

    def __init__(
        self,
        model,
        optimizer_forward,
        optimizer_backward,
        train_config,
    ):

        super().__init__(
            model,
            optimizer_forward,
            optimizer_backward,
            train_config,
        )

    def train(
        self,
        train_loader: DataLoader,
    ):

        for imf_iteration in range(self.n_imf):

            self.imf_iteration = imf_iteration

            # BACKWARD
            for refresh_iteration in range(self.n_refresh):

                new_train_loader_backward = self._generate_new_train_loader_backward(train_loader)

                for epoch in range(self.n_epoch):

                    for x0, x1 in new_train_loader_backward:
                        self.optimizer_backward.zero_grad()
                        self.train_step_backward(x0, x1)

            # FORWARD
            for refresh_iteration in range(self.n_refresh):

                new_train_loader_forward = self._generate_new_train_loader_forward(train_loader)

                for epoch in range(self.n_epoch):

                    for x0, x1 in new_train_loader_forward:
                        self.optimizer_forward.zero_grad()
                        self.train_step_forward(x0, x1)

    def train_step_forward(
        self,
        x0: TensorType,
        x1: TensorType,
    ):

        t = torch.rand_like(x0)
        z = torch.randn_like(x0)

        loss_forward = self.model.loss_forward(x0, x1, z, t)

        loss_forward.backward()

        self.optimizer_forward.step()

    def train_step_backward(
        self,
        x0: TensorType,
        x1: TensorType,
    ):

        t = torch.rand_like(x0)
        z = torch.randn_like(x0)

        loss_backward = self.model.loss_backward(x0, x1, z, t)

        loss_backward.backward()

        self.optimizer_backward.step()

    def _generate_new_train_loader_backward(
        self,
        train_loader: DataLoader,
    ) -> DataLoader:
        # TODO change channel
        new_tensor_x0 = torch.zeros((len(train_loader.dataset), 1, *self.img_size))
        new_tensor_x1 = torch.zeros((len(train_loader.dataset), 1, *self.img_size)) 

        index = 0
        if self.imf_iteration != 0:
            for x0, _ in train_loader:

                x1 = self.model.sample_forward_sde(x0, self.n_timesteps)

                new_tensor_x0[index: index + x0.shape[0], ...] = x0
                new_tensor_x1[index: index + x0.shape[0], ...] = x1

                index += x0.shape[0]
                
            new_tensor_dataset_x0 = TensorDataset(new_tensor_x0)
            new_tensor_dataset_x1 = TensorDataset(new_tensor_x1)
            
            new_tensor_dataset = StackDataset([new_tensor_dataset_x0, new_tensor_dataset_x1])
            
            new_train_loader = DataLoader(new_tensor_dataset, batch_size = train_loader.batch_size)
        else:
            new_train_loader = train_loader

        return new_train_loader

    def _generate_new_train_loader_forward(
        self,
        train_loader
    ) -> DataLoader:

        new_tensor = torch.zeros((len(train_loader.dataset), 2, 1, *self.img_size))

        index = 0
        for _, x1 in train_loader:

            x0 = self.model.sample_backward_sde(x1, self.n_timesteps)

            new_tensor[index: index + x0.shape[0], 0, ...] = x0
            new_tensor[index: index + x0.shape[0], 1, ...] = x1

            index += x0.shape[0]
        new_tensor_dataset = TensorDataset(new_tensor)
        new_train_loader = DataLoader(new_tensor_dataset, batch_size = self.batch_size)

        return new_train_loader
