import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from pandas import DataFrame

def process_csv(path: 'Path | str'):
    df = pd.read_csv(str(path))
    return process_df(df)

def process_df(df: 'DataFrame'):
    df = df[["Text","Category"]].dropna()        
    return df.Text.to_numpy(), df.Category.to_numpy()