import json

from sklearn.model_selection import train_test_split

from .logger import logging as log
from .pipeline.ingestion import process_csv, process_df
from .pipeline.model import NewsClassifier
from .paths import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from pandas import DataFrame

def train_model(data: 'Path | str | DataFrame', test_size: float, model_path: 'Path | str'=MODEL_PATH):
    log.info("Model training started...")
    
    model = NewsClassifier()
    
    X, Y = None, None
    if isinstance(data, str) or isinstance(data, Path):
       X, Y = process_csv(data)
    else: X, Y = process_df(data) 
    
    X_train, X_test, Y_train, Y_test = [None] * 4
    
    if test_size==0: X_train, Y_train = X, Y
    else: X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    model.fit(X_train, Y_train)
    model.save(model_path)
    
    if test_size!=0: 
        scores = model.evaluate(X_test, Y_test)    
        with open(Path(model_path).parent / "eval_scores.json", 'w') as f:
            json.dump(scores, f, indent=4)
        
    return model

if __name__=="__main__":
    train_model(DATASET_PATH / "BBC News Train.csv", 0)