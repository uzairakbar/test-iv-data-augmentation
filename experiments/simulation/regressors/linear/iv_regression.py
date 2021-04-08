import numpy as np
from sklearn.linear_model import LinearRegression
from data_augmentor import augment_data

class IVR(LinearRegression):
    def __init__(self, **kwargs):
        super(IVR, self).__init__()

    def fit(self, X, y, augment_features = 0, lamda = 1.0, **kwargs):
        X_augmented, g = augment_data(X, augment_features)
        
        PIz = g @ np.linalg.pinv(g)
        if lamda < 1.0:
            gamma = (1/1-lamda)**0.5 - 1.0
            multiplier = ( np.eye( len(X) ) + gamma*PIz )
        else:
            multiplier = PIz
        return super(IVR, self).fit( multiplier @ X_augmented, multiplier @ y)
