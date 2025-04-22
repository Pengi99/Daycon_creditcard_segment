from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RFModel(BaseModel):
    def __init__(self, **params):
        super().__init__(params)
        self.model = RandomForestClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)