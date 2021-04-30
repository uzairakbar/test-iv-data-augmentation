import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel

class MMRIV(RegressorMixin):
    def __init__(self, alpha = 1.0, kernel = "rbf", gamma = None, **kwargs):
        super(MMRIV, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.kernel = kernel

    def fit(self, X, y, augment_features = 0, lamda = 1.0, **kwargs):
        X_augmented, Z = augment_data(X, augment_features)
        self.X = X_augmented

        if self.gamma is None:
            gamma = 1.0/X_augmented.shape[-1]
        else:
            gamma = self.gamma

        n = len(X_augmented)
        if self.kernel == "rbf":
            K = rbf_kernel(Z, gamma = gamma)/(n**2)
            L = rbf_kernel(X_augmented)
        else:
            raise NotImplementedError
        Ktilde = lamda*K + (1-lamda)*np.eye(n)
        self.W = np.linalg.pinv( L @ Ktilde @ L + self.alpha*L ) @ L @ Ktilde @ y

        return self
    
    def predict(self, X):
        if self.kernel == "rbf":
            L = rbf_kernel(X, self.X)
        else:
            raise NotImplementedError
        
        return L @ self.W
    
    def get_params(self, deep=True):
        out = dict()
        out["gamma"] = self.gamma
        out["alpha"] = self.alpha
        out["kernel"] = self.kernel
        return out
    
    def set_params(self, **params):
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.kernel = params["kernel"]
        return self
