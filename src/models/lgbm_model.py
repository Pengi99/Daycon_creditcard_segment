import lightgbm as lgb
from .base import BaseModel

class LGBMModel(BaseModel):
    def __init__(self, **params):
        super().__init__(params)
        self.model = lgb.LGBMClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)