import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader, TensorDataset

from medical_imaging_imf.models.schrodinger_bridge import SchrodBridgeIMF


class Trainer():

    def __init__(
        self,
        model: SchrodBridgeIMF,
        optimizer_forward,
        optimizer_backward,
        train_config,
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

            for refresh_iteration in range(self.n_refresh):

                # NOTE : could load old model weights here
                new_train_dataloader = self._generate_new_train_loader(train_loader)

                for epoch in range(self.n_epoch):

                    for x0, x1 in new_train_dataloader:

                        self.optimizer_forward.zero_grad()
                        self.optimizer_backward.zero_grad()

                        self.train_step(x0, x1)


    def train_step(
        self,
        x0: TensorType,
        x1: TensorType,
    ):

        t = torch.rand_like(x0)
        z = torch.randn_like(x0)

        loss = (
            self.model.loss_forward(x0, x1, z, t)
            + self.model.loss_backward(x0, x1, z, t)
            + self.lambda * self.model.loss_consistency(x0, x1, z, t)
        )

        loss.backward()

        self.optimizer_forward.step()
        self.optimizer_backward.step()


    def _generate_new_train_loader(
        self,
        train_loader: DataLoader,

    ):
        new_tensor = torch.zeros((len(train_loader.dataset), 2, 1, *self.img_size))

        index = 0
        if self.imf_iteration != 0:
            for x0, x1 in train_loader:

                x0, x1 = self.model.mixture(x0, x1)

                new_tensor[index: index + x0.shape[0], 0, ...] = x0
                new_tensor[index: index + x0.shape[0], 1, ...] = x1

                index += x0.shape[0]
            new_tensor_dataset = TensorDataset(new_tensor)
            new_train_loader = DataLoader(new_tensor_dataset, batch_size = self.batch_size)
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
        new_tensor = torch.zeros((len(train_loader.dataset), 2, 1, *self.img_size))

        index = 0
        if self.imf_iteration != 0:
            for x0, _ in train_loader:

                x1 = self.model.sample_forward_sde(x0, self.n_timesteps)

                new_tensor[index: index + x0.shape[0], 0, ...] = x0
                new_tensor[index: index + x0.shape[0], 1, ...] = x1

                index += x0.shape[0]
            new_tensor_dataset = TensorDataset(new_tensor)
            new_train_loader = DataLoader(new_tensor_dataset, batch_size = self.batch_size)
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
