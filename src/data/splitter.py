import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_kfold_split(X, y, n_splits=5, random_state=42, shuffle=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        yield fold, tr_idx, val_idx


def train_val_split(X, y, test_size=0.3, random_state=42, stratify=True):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=(y if stratify else None)
    )