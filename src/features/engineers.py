import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import VarianceThreshold

class FeatureEngineer:
    """
    Preprocess loaded raw data: drop high-NA cols, impute, merge to train/test DataFrames.
    Optionally filter by selected features list.
    """
    def __init__(self, months=None, na_ratio=0.2, select_csv=None, slice_n=1, random_state=42):
        self.months = months or ['07','08','09','10','11','12']
        self.na_ratio = na_ratio
        self.select_csv = select_csv
        self.slice_n = slice_n
        self.random_state = random_state

    def _merge(self, dfs: dict, split: str) -> pd.DataFrame:
        order = ["customer","credit","sales","billing","balance","channel","marketing","performance"]
        base = dfs.get(f"customer_{split}_df")
        if base is None:
            raise ValueError("Missing customer base DataFrame for merge")
        for p in order[1:]:
            key = f"{p}_{split}_df"
            if key not in dfs:
                continue
            df = dfs[key]
            if not {'기준년월','ID'}.issubset(df.columns):
                print(f"[WARN] {key} missing merge keys, skipped")
                continue
            base = base.merge(df, on=['기준년월','ID'], how='left')
        print(f"{split.upper()} merged shape: {base.shape}")
        return base

    def preprocess(self, loaded: dict, select_features=False):
        cats = ["customer","credit","sales","billing","balance","channel","marketing","performance"]
        train_dfs, test_dfs = {}, {}
        # TRAIN processing
        for p in cats:
            chunks = []
            for m in self.months:
                key = f"{p}_train_{m}"
                if key in loaded:
                    dfm = loaded[key]
                    # Monthly sampling to reduce memory
                    if self.slice_n > 1:
                        dfm = dfm.sample(frac=1/self.slice_n, random_state=self.random_state)
                    chunks.append(dfm)
            if chunks:
                df = pd.concat(chunks, axis=0)
                # drop high-NA columns
                df = df.drop(columns=df.columns[df.isna().mean() > self.na_ratio])
                # impute
                for c in df.columns:
                    if df[c].isna().any():
                        if is_numeric_dtype(df[c]):
                            df[c].fillna(df[c].median(), inplace=True)
                        else:
                            df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else 'MISSING', inplace=True)
                train_dfs[f"{p}_train_df"] = df
        train_df = self._merge(train_dfs, 'train')
        # Remove highly imbalanced categorical features
        cat_threshold = 0.99
        imbalanced_cols = []
        n_train = len(train_df)
        for col in train_df.select_dtypes(include=['object', 'category', 'int64']).columns:
            freqs = train_df[col].value_counts(dropna=False)
            if freqs.iloc[0] / n_train >= cat_threshold:
                imbalanced_cols.append(col)
        if imbalanced_cols:
            print(f"[INFO] Dropping imbalanced categorical columns: {imbalanced_cols}")
            train_df.drop(columns=imbalanced_cols, inplace=True)

        # Remove low variance numeric features
        var_threshold = 1e-4
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        selector = VarianceThreshold(threshold=var_threshold)
        selector.fit(train_df[num_cols])
        low_var_cols = num_cols[~selector.get_support()]
        if len(low_var_cols) > 0:
            print(f"[INFO] Dropping low variance numeric columns: {list(low_var_cols)}")
            train_df.drop(columns=list(low_var_cols), inplace=True)
        # TEST processing (mirror train)
        for p in cats:
            chunks = []
            for m in self.months:
                key = f"{p}_test_{m}"
                if key in loaded:
                    dfm = loaded[key]
                    # Monthly sampling to reduce memory
                    if self.slice_n > 1:
                        dfm = dfm.sample(frac=1/self.slice_n, random_state=self.random_state)
                    chunks.append(dfm)
            if chunks:
                df = pd.concat(chunks, axis=0)
                # align columns
                extra = [c for c in df.columns if c not in train_df.columns]
                if extra:
                    df.drop(columns=extra, inplace=True)
                test_dfs[f"{p}_test_df"] = df
        test_df = self._merge(test_dfs, 'test')
        # impute test
        for c in test_df.columns:
            if test_df[c].isna().any():
                if is_numeric_dtype(test_df[c]):
                    test_df[c].fillna(test_df[c].median(), inplace=True)
                else:
                    test_df[c].fillna(test_df[c].mode().iloc[0] if not test_df[c].mode().empty else 'MISSING', inplace=True)
        # feature selection
        if select_features and self.select_csv:
            sel = pd.read_csv(self.select_csv)['feature'].tolist()
            keep = [c for c in sel if c in train_df.columns] + ['기준년월','ID','Segment']
            train_df = train_df[keep]
            test_df = test_df[[c for c in keep if c in test_df.columns]]
        return train_df, test_df