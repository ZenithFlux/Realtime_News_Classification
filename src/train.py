from .logger import logging as log
from .pipeline.data import ClassifierDataset
from .pipeline.model import NewsClassifier
from .paths import *

def main():
    log.info("Model training started...")
    data = ClassifierDataset().process_csv(DATASET_PATH / "train.csv", 0)
    model = NewsClassifier()
    
    model.fit(data)
    model.save(MODEL_PATH)

if __name__=="__main__":
    main()