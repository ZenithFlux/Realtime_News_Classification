import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from pandas import DataFrame
    from numpy import ndarray

def process_csv(path: 'Path | str') -> 'tuple[ndarray, ndarray]':
    """
    Read a csv file and return ndarray of article text and catgory labels.
    The csv must have columns 'Text' and 'Category' to extract articles
    and labels respectively.
    """
    
    df = pd.read_csv(str(path))
    return process_df(df)

def process_df(df: 'DataFrame') -> 'tuple[ndarray, ndarray]':
    """
    Read a pandas Dataframe and return ndarray of article text and catgory labels.
    The dataframe must have columns 'Text' and 'Category' to extract articles
    and labels respectively.
    """
    
    df = df[["Text","Category"]].dropna()        
    return df.Text.to_numpy(), df.Category.to_numpy()