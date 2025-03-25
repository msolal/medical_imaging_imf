import torch
from torchtyping import TensorType

class ShrodBridgeIMF(torch.nn.Module):

    def __init__(self,
                network_forward : torch.nn.Module,
                network_backward : torch.nn.Module,
                ema : torch.nn.Module,
                sigma : float = 1.
                ):
        
        self.network_forward = network_forward
        self.network_backward = network_backward

        self.sigma = torch.tensor(sigma)
    
    def sample_Q_t_01(self,
                x0: TensorType["batch","channels","width","height"],
                x1 : TensorType["batch","channels","width","height"],
                z: TensorType["batch","channels","width","height"],
                t : TensorType["batch","channels","width","height"]
                ) -> TensorType["batch","channels","width","height"]:
        """
        joint probability \Pi^{0}_{t|0,1} = Q_{t|0,1}
        """

        return t * x1 + (1-t)* x0 + self.sigma * torch.sqrt(t*(1-t)) * z

    @torch.no_grad()
    def forward_sample_sde(self,
                        x : TensorType["batch","channels","width","height"],
                        N: int,
                        device : str
                        ):
        """
        Perform a forward SDE sampling process.
        """
        dt = 1./N
        batch_size = x.shape[0]
        ts = torch.arange(N)/N
        trajectories = []
        for i in range(N):
            t = torch.ones((batch_size,1), device = device) * ts[i]
            x = x.detach().clone() + self.network_forward(t,x) * dt + self.sigma * torch.randn_like(x) * torch.sqrt(dt)
            trajectories.append(x.detach().clone())
    
    @torch.no_grad()
    def backward_sample_sde(self,
                        x : TensorType["batch","channels","width","height"],
                        N: int,
                        device : str
                        ):
        """
        Perform a backward SDE sampling process.
        """
        dt = 1./N
        batch_size = x.shape[0]
        ts = 1 - torch.arange(N)/N
        trajectories = []
        for i in range(N):
            t = torch.ones((batch_size,1), device = device) * ts[i]
            x = x.detach().clone() + self.network_forward(t,x) * dt + self.sigma * torch.randn_like(x) * torch.sqrt(dt)
            trajectories.append(x.detach().clone())


    def loss_forward(self,
                    x0: TensorType["batch","channels","width","height"],
                    x1 : TensorType["batch","channels","width","height"],
                    z: TensorType["batch","channels","width","height"],
                    t : TensorType["batch","channels","width","height"]
                    ) -> TensorType["batch","channels","width","height"]:
        """
        nabla log Q_{T|t}(X_T | X_t) = (X_1 - X_t)/(1-t) then replacing x_t by the interpolant 
        we use the scaling of the loss Appendix H
        """
        x_0_1_t = self.sample_Q_t_01(x0,x1,z,t)

        loss_scaling = (1 + self.sigma.pow(2)*(t)/(1-t)).pow(-1)
        
        loss = loss_scaling * (
            (x1 - x0 - self.sigma* torch.sqrt(t/(1-t)) * z - self.network_forward(t,x_0_1_t))
            .square().sum(0))
        return loss.sum()
    
    def loss_backward(self,
                    x0: torch.Tensor,
                    x1 : torch.Tensor,
                    z: torch.Tensor,
                    t : torch.Tensor
                    ) -> torch.Tensor:
        """
        nabla log Q_{t|0}(X_t | X_0) = (X_0 - X_t)/(t) then replacing x_t by the interpolant 
        we use the scaling of the loss Appendix H
        """
        x_0_1_t = self.sample_Q_t_01(x0,x1,z,t)

        loss_scaling = (1 + self.sigma.pow(2)*(1-t)/(t)).pow(-1)
        
        loss = loss_scaling * ( x0 - x1 - self.sigma* torch.sqrt((1-t)/t) * z - self.network_backward(t,x_0_1_t)).square().sum(0)
        return loss.sum()
    

class TrainerIMF:

    def __init__(self,
                 shrod_bridge_imf,
                 n_iteration :int,
                 n_inner_iteration : int
                 ):
        
        self.n_iteration = n_iteration
        self.n_inner_iteration = n_inner_iteration

        self.shrod_bridge_imf = shrod_bridge_imf
        


   