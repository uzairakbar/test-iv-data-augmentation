from sklearn.base import RegressorMixin
from .linear.iv_regression import IVR
from .nonlinear.nn.model import NN
from .nonlinear.mmriv.mmriv import NN

class Regressor(RegressorMixin):
    def __init__(self, regression = "linear", **kwargs):
        super(Regressor, self).__init__()
        self.regression = regression
        if self.regression == "linear":
            self.regressor = IVR(**kwargs)
        elif  self.regression == "nn":
            self.regressor = NN(**kwargs)
        elif  self.regression == "mmriv":
            parameters = {'kernel':['rbf'],
              'alpha': alpha_range.rvs(random_state=seed, size=5),
              'gamma': gamma_range.rvs(random_state=seed, size=5)}
            self.regressor = GridSearchCV(MMRIV(), parameters)

    def fit(self, X, y, augment_features = 0, lamda = 0.0, **kwargs):
        return self.regressor.fit(X, y, augment_features, lamda, **kwargs)
    
    def predict(self, X):
        return self.regressor.predict(X)
