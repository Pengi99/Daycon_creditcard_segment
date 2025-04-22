from catboost import CatBoostClassifier
from .base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, **params):
        super().__init__(params)
        self.model = CatBoostClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y, verbose=False)

    def predict(self, X):
        return self.model.predict(X)