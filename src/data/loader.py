import pandas as pd
import gc
from tqdm.auto import tqdm

class DataLoader:
    """
    Load raw parquet data for train/test and keep in a dict.
    """
    def __init__(self, months=None, data_dir='./'):
        self.months = months or ['07','08','09','10','11','12']
        self.data_dir = data_dir
        self.loaded_data = {}

    def load(self):
        cats = {
            "customer": ("1.회원정보","회원정보","customer"),
            "credit":  ("2.신용정보","신용정보","credit"),
            "sales":   ("3.승인매출정보","승인매출정보","sales"),
            "billing": ("4.청구입금정보","청구정보","billing"),
            "balance": ("5.잔액정보","잔액정보","balance"),
            "channel": ("6.채널정보","채널정보","channel"),
            "marketing":("7.마케팅정보","마케팅정보","marketing"),
            "performance":("8.성과정보","성과정보","performance"),
        }
        print("▶ Loading raw data...")
        for split in ["train","test"]:
            for folder, suffix, prefix in cats.values():
                for m in self.months:
                    path = f"{self.data_dir}/{split}/{folder}/2018{m}_{split}_{suffix}.parquet"
                    key = f"{prefix}_{split}_{m}"
                    try:
                        self.loaded_data[key] = pd.read_parquet(path)
                    except FileNotFoundError:
                        continue
        gc.collect()
        print(f"✔ Loaded {len(self.loaded_data)} tables")
        return self.loaded_data