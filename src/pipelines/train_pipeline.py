import yaml
import pandas as pd
from src.data.loader import DataLoader
from src.features.engineers import FeatureEngineer
from src.data.splitter import train_val_split
from src.models.factory import ModelFactory
from src.utils.metrics import METRICS
from src.utils.result_manager import save_predictions_and_params


class TrainPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open('configs/models.yaml') as f:
            self.model_cfg = yaml.safe_load(f)
        self.loader = DataLoader(
            months=self.cfg['data'].get('months'),
            data_dir=self.cfg['data']['path_dir']
        )
        self.fe = FeatureEngineer(
            months=self.cfg['data'].get('months'),
            na_ratio=self.cfg['features']['na_ratio'],
            select_csv=self.cfg['features'].get('select_csv')
        )

    def run(self):
        # 1. Load raw tables
        raw = self.loader.load()
        # 2. Preprocess & merge
        train_df, test_df = self.fe.preprocess(
            raw,
            select_features=self.cfg['features'].get('select', False)
        )
        X = train_df.drop(['ID','Segment'], axis=1)
        y = train_df['Segment']
        # 3. Train/Val split
        X_tr, X_val, y_tr, y_val = train_val_split(
            X, y,
            test_size=self.cfg['split']['test_size'],
            random_state=self.cfg['split']['random_state']
        )
        # 4. Model
        # 2) Load model name and parameters from separate models.yaml
        model_name = self.model_cfg['pipeline']['model_name']
        params = self.model_cfg['models'][model_name]
        model = ModelFactory.create(model_name, params)
        model.train(X_tr, y_tr)
        results = model.evaluate(X_val, y_val, METRICS)
        print("Validation:", results)
        # 5. Retrain on full and predict
        model.train(X, y)
        preds = model.predict(test_df.drop('ID', axis=1))
        out = pd.DataFrame({'ID': test_df['ID'], 'Segment': preds})
        out = out.groupby('ID')['Segment'].agg(lambda s: s.value_counts().idxmax()).reset_index()
        self.out = out
        out.to_csv(self.cfg['output']['file_name'], index=False)
        print(f"Saved predictions to {self.cfg['output']['file_name']}")

    def save_result(self):
        model_name = self.model_cfg['pipeline']['model_name']
        params = self.model_cfg['models'][model_name]
        save_predictions_and_params(
            model_name=model_name,
            preds_df=self.out,
            params=params,
            result_base=self.cfg['output']['result_dir']
        )