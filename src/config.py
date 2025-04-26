from pathlib import Path

raw_data_path = Path("data/raw/Reviews.csv")
preprocessed_data_path = Path("data/processed/processed.csv")
model_saving_path = Path("models/model.pkl")

# Ensure folders exist, not files
raw_data_path.parent.mkdir(parents = True, exist_ok = True)
preprocessed_data_path.parent.mkdir(parents = True, exist_ok = True)
model_saving_path.parent.mkdir(parents = True, exist_ok = True)
