# Read raw data and prepares for preprocessing (extracting data from CSV, databases, API, etc.) 

import pandas as pd

def load_data(path):
    return pd.read_csv(path) 