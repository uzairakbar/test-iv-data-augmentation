import numpy as np
from torch.utils.data import Dataset

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

class dataset(Dataset):
    'SEM'
    def __init__(self,
                 x,
                 y,
                 augment_features = 0):
        'Initialization'
        super(dataset, self).__init__()
        self.y = y
        self.x, self.g = augment_data(x, augment_features)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x[index, :]
        g = self.g[index, :]
        y = self.y[index:index+1]

        # convert to tensor
        x = torch.from_numpy(x)
        g = torch.from_numpy(g)
        y = torch.from_numpy(y)

        return x, y, g
