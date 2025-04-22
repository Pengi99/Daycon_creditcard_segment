from .lgbm_model import LGBMModel
from .catboost_model import CatBoostModel
from .rf_model import RFModel

MODEL_REGISTRY = {
    'lgbm': LGBMModel,
    'catboost': CatBoostModel,
    'rf': RFModel,
}

class ModelFactory:
    @staticmethod
    def create(name: str, params: dict):
        cls = MODEL_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown model: {name}")
        return cls(params)