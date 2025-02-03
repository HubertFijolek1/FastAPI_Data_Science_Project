import pandas as pd

def load_and_clean_data(file_path: str):
    """
    Loads data from a CSV, drops duplicates, fills missing values forward.
    """
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def load_and_clean_data_from_df(df: pd.DataFrame):
    """
    Cleans an existing DataFrame similarly.
    """
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def analyze_data(df: pd.DataFrame):
    """
    Returns descriptive statistics of the DataFrame.
    """
    return df.describe()
