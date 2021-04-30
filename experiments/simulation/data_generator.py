import torch

class SEM(object):
    def __init__(self,
                 dim,
                 sem = "linear",
                 transform_h = False,
                 additional_noise_y = False,
                 transform_x1_and_x2 = False,
                 additional_noise_x1_and_x2 = False):
        """
        Parameters
        ----------
        dim : int
            Dimension of data samples X
        sem : str, optional
            Linear vs non-linear generating model
        transform_h : boolean, potional
            Generate random transform Whx and Why from confounder h to x and y respectively
        transform_x1_and_x2 : boolean, optional
            Generate random transform Wxy from confounder x to y
        additional_noise_x1_and_x2 : boolean, optional
            Add noise to covariates x
        additional_noise_y : boolean, optional
            Add noise to labels y
        """
        self.transform_h = transform_h
        self.transform_x1_and_x2 = transform_x1_and_x2
        self.additional_noise_x1_and_x2 = additional_noise_x1_and_x2
        self.additional_noise_y = additional_noise_y

        # dim is the number of dimensions of x
        self.dim_half = dim//2

        # Linear transformation h to y and x
        if transform_h:
            self.Why = torch.randn(self.dim_half, self.dim_half) / dim
            self.Whx1 = torch.randn(self.dim_half, self.dim_half) / dim
            self.Whx2 = torch.randn(self.dim_half, self.dim_half) / dim
        else:
            self.Why = torch.eye(self.dim_half)
            self.Whx1 = torch.eye(self.dim_half)
            self.Whx2 = torch.eye(self.dim_half)

        # scaler to make sure slope along x1 is lower than x2
        if sem == 'linear':
            scaler = 2.0
        else:
            scaler = 1.0
        
        # Linear transformation from x to y (adjust slopes)
        if transform_x1_and_x2:
            self.Wx2y = torch.randn(self.dim_half, self.dim_half)*scaler / dim
            self.Wx1y = torch.randn(self.dim_half, self.dim_half) / (dim*scaler)
        else:
            self.Wx2y = torch.eye(self.dim_half)*scaler
            self.Wx1y = torch.eye(self.dim_half)/scaler
        
        self.sem = sem

    def __call__(self, 
                 N, 
                 train=True, 
                 confounding = True):
        # Add noise to x
        if self.additional_noise_x1_and_x2:
            noise_x1 = torch.randn(N, self.dim_half)*0.1
            noise_x2 = torch.randn(N, self.dim_half)*0.1
        else:
            noise_x1 = 0
            noise_x2 = 0

        # Add noise to y
        if self.additional_noise_y:
            noise_y = torch.randn(N, self.dim_half)*0.1
        else:
            noise_y = 0

        # Sample values for confounder and covariates
        if train:
            h = torch.randn(N, self.dim_half)
            x1 = h @ self.Whx1 + noise_x1
            x2 = h @ self.Whx2 + noise_x2
        else:
            h = torch.randn(N, self.dim_half)*0.1
            x1 = h @ self.Whx1 + noise_x1
            x2 = h @ self.Whx2 + noise_x2

        # Compute y
        if confounding:
            if self.sem == "linear":
                y = x1 @ self.Wx1y + x2 @ self.Wx2y + h @ self.Why + noise_y
            else:
                y = x2 @ self.Wx2y + torch.sigmoid(x1 @ self.Wx1y) + h @ self.Why + noise_y
        else:
            # dox1 = torch.randn(N, self.dim_half)
            # dox2 = torch.randn(N, self.dim_half)
            h_ = torch.randn(N, self.dim_half)
            dox1 = h_ @ self.Whx1 + noise_x1
            dox2 = h_ @ self.Whx2 + noise_x2
            if self.sem == "linear":
                y = dox1 @ self.Wx1y + dox2 @ self.Wx2y + h @ self.Why + noise_y
            else:
                y = dox2 @ self.Wx2y + torch.tanh(dox1 @ self.Wx1y) + h @ self.Why + noise_y
            

        return torch.cat((x1, x2), dim=1).numpy(), y.sum(dim=1).numpy(), h.sum(dim=1).numpy()
