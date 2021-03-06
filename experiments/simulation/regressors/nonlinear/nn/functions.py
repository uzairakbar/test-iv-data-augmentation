import torch
from torch.nn import functional as F

def iv_loss(y_pred, y, z, lamda=1.0):

    PIz = z.mm( torch.pinverse(z) )
    if lamda < 1.0:
        batch_size = z.size(0)
        gamma = (1/1-lamda)**0.5 - 1.0
        multiplier = ( torch.eye( batch_size ) + gamma*PIz )
    else:
        multiplier = PIz
    
    loss = F.mse_loss(multiplier.mm( y_pred ), multiplier.mm(y), reduction="mean")

    return loss
