import torch
from torch import nn

class LinearBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 last = False):
        super(LinearBlock, self).__init__()
        modules = [nn.Linear(in_channel, out_channel)]
        if not last:
            modules += [nn.BatchNorm1d(out_channel),
                        nn.LeakyReLU(inplace=True)]
        self.block = nn.Sequential(*modules)
    def forward(self, x):
        return self.block(x)

class FCNN(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 width = 5,
                 layers = 2):
        super(FCNN, self).__init__()
        hidden = [LinearBlock(in_channel, width)]
        for layer in range(layers - 1):
            hidden += [LinearBlock(width, width)]
        self.hidden = nn.Sequential(*hidden)
        self.out = LinearBlock(width, out_channel, last=True)
    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .misc import plot_loss_graphs
from .data_utils import dataset
from .solver import solver

ARGS = {"in_dim" : 10,
        "out_dim" : 1,
        "epochs" : 150,
        "device" : "cpu",
        "batch_size" : 256,
        "print_logs" : False,
        "log_interval" : 100,
        "train_val_split" : 0.2,
        "lr" : 0.0001,
        "width" : 5,
        "layers" : 3,
        "optimizer" : "adam"}

class NN(RegressorMixin):
    def __init__(self, in_dim, out_dim, width = 5, layers = 2, **kwargs):
        super(NN, self).__init__()
        self.model = FCNN(in_dim, out_dim, width, layers)
        self.train_logs, self.test_logs = [], []

    def fit(self, X, y, augment_features = 0, lamda = 0.0, **kwargs):
        if not kwargs:
            kwargs = ARGS
        
        if kwargs["train_val_split"]:
            X_train, X_val, y_train, y_val = train_test_split( X, y, test_size = kwargs["train_val_split"] )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        train_dataset = dataset(X_train, y_train, augment_features)
        val_dataset = dataset(X_val, y_val, augment_features)

        train_loader = DataLoader(train_dataset, batch_size = kwargs["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = kwargs["batch_size"], shuffle=True)

        self.model, self.train_logs, self.test_logs = solver(model = self.model,
                                                             train_loader = train_loader, 
                                                             test_loader = val_loader,
                                                             lamda = lamda, 
                                                             **kwargs)
        self.model.eval()
        
        if kwargs["print_logs"]:
            plot_loss_graphs((self.train_logs, self.test_logs), kwargs);

    def predict(self, X):
        y = self.model(torch.tensor(X)).detach().numpy()
        return y
