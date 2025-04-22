from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, **params: dict):
        self.params = params

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y, metrics: dict):
        preds = self.predict(X)
        return {name: fn(y, preds) for name, fn in metrics.items()}