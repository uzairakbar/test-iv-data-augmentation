import torch
import numpy as np

def augment_data(x, pert, U=0.1, model = "linear"):
    """
    Parameters
    ----------
    x : np.array
        Feature vector to augment with additive noise
    pert : int
        Number of features to augment
    U : float, optional
        The range [-U, U] for uniform additive noise as data augmentation
    """
    if model == "linear":
        # Add uniform noise ~ [-U, U] to features of x
        noise_x = np.zeros_like(x)
        noise_x[:, :x.shape[1]//2] = torch.FloatTensor(x.shape[0], x.shape[1]//2).uniform_(-U, U).numpy()
        feature_index_no_noise = np.random.choice(x.shape[1]//2, x.shape[1]//2 - pert, replace=False)
        noise_x[:, feature_index_no_noise] = 0.0
        x += noise_x
        g = noise_x
    elif model == "nonlinear":
#         g = np.random.choice([-1.0, 1.0], p=[0.5, 0.5])
#         if g > 0:
#             x *= U
        g = np.ones_like(x)
        g[:, :x.shape[1]//2] = torch.FloatTensor(x.shape[0], x.shape[1]//2).uniform_(0, U).numpy()
        feature_index_no_noise = np.random.choice(x.shape[1]//2, x.shape[1]//2 - pert, replace=False)
        g[:, feature_index_no_noise] = 1.0
        x = x*g
    return x, g
