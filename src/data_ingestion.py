from pathlib import Path
import pandas as pd

def load_data(path: str, use_drive: bool = False) -> pd.DataFrame:
    if use_drive:
        import gdown
        data_save_path = Path("data/raw/Reviews.csv")
        data_save_path.parent.mkdir(parents = True, exist_ok = True)
        gdown.download(id = path, output = str(data_save_path), quiet = False)
        return pd.read_csv(data_save_path)
    else:
        return pd.read_csv(path)
