import json

from .logger import logging as log
from .pipeline.data import ClassifierDataset
from .pipeline.model import NewsClassifier
from .paths import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from pandas import DataFrame

def train_model(data: 'Path | str | DataFrame', test_size: float, model_path: 'Path | str'=MODEL_PATH):
    log.info("Model training started...")
    
    model = NewsClassifier()
    dataset = ClassifierDataset()
    
    if isinstance(data, str) or isinstance(data, Path):
       dataset.process_csv(data, test_size)
    else: dataset.process_df(data, test_size) 
    
    model.fit(dataset.X_train, dataset.Y_train)
    scores = model.evaluate(dataset.X_test, dataset.Y_test)
    
    model.save(model_path)
    with open(Path(model_path).parent / "eval_scores.json", 'w') as f:
        json.dump(scores, f, indent=4)
        
    return model

if __name__=="__main__":
    train_model(DATASET_PATH / "BBC News Train.csv", 0.1)