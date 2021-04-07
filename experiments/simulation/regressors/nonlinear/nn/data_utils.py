import numpy as np
from torch.utils.data import Dataset
from ....data_augmentor import augment_data

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
