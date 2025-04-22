import yaml
import pandas as pd
from src.data.loader import DataLoader
from src.features.engineers import FeatureEngineer
from sklearn.ensemble import RandomForestClassifier

class FeatureSelectionPipeline:
    """
    랜덤포레스트 중요도 + 상관도 필터를 이용해
    selected_features.csv를 생성합니다.
    """
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.loader = DataLoader(
            months=self.cfg['data']['months'],
            data_dir=self.cfg['data']['path_dir']
        )
        self.fe = FeatureEngineer(
            months=self.cfg['data']['months'],
            na_ratio=self.cfg['features']['na_ratio'],
            slice_n=self.cfg['select'].get('slice_n', 1),
            random_state=self.cfg['select'].get('random_state', 42)
        )

    def run(self):
        # 1) 로드 & 합치기
        raw = self.loader.load()
        train_df, _ = self.fe.preprocess(raw, select_features=False)

        # 2) 간단한 레이블 인코딩
        X = train_df.drop(['ID', 'Segment', '기준년월'], axis=1)
        y = train_df['Segment']
        for col in X.select_dtypes(include='object').columns:
            X[col] = pd.factorize(X[col])[0]

        # 3) RF로 중요도 계산
        rf = RandomForestClassifier(**self.cfg['select']['rf_params'])
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=X.columns)\
                .sort_values(ascending=False)

        # 4) 상위 top_n + 상관도 필터링
        top_feats = imp.head(self.cfg['select']['top_n']).index.tolist()
        corr = X[top_feats].corr().abs()
        selected = []
        for f in top_feats:
            if all(corr.loc[f, s] <= self.cfg['select']['corr_threshold']
                   for s in selected):
                selected.append(f)
        # 반드시 포함할 피처 (중복 없이)
        mandatory = self.cfg['select'].get(
            'mandatory_features',
            ['기준년월', 'ID', 'Segment']
        )
        selected = list(dict.fromkeys(selected + mandatory))

        # 5) CSV로 저장
        out_csv = self.cfg['select']['output_csv']
        pd.DataFrame(selected, columns=['feature']).to_csv(out_csv, index=False)
        print(f"✔ {len(selected)}개 피처 선택 완료 → {out_csv}")