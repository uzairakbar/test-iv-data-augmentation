import torch
import numpy as np

def augment_data(x, pert, U=1):
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
    # Add uniform noise ~ [-U, U] to features of x
    noise_x = np.zeros_like(x)
    noise_x[:, :x.shape[1]//2] = torch.FloatTensor(x.shape[0], x.shape[1]//2).uniform_(-U, U).numpy()
    feature_index_no_noise = np.random.choice(x.shape[1]//2, x.shape[1]//2 - pert, replace=False)
    noise_x[:, feature_index_no_noise] = 0.0
    x += noise_x
    return x, noise_x
