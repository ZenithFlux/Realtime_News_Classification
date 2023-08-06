import re

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split

from .logger import logging as log

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
    from pathlib import Path


class ClassifierDataset:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.X_train = self.Y_train = self.X_test = self.Y_test = None
        
    def read_csv(self, path: 'Path' | str, test_size=0.2):
        df = pd.read_csv(str(path))
        self.set_data(df, test_size)
    
    def set_data(self, df: 'DataFrame', test_size: float=0.2):
        df = df[["Text","Category"]].dropna()
        
        X = df.Text.apply(self.preprocess).to_numpy()
        Y = df.Category.to_numpy()
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size)
        log.info("Data Transformation completed...")
        
    def preprocess(self, text: str):
        doc = self.nlp(text)
        processed_text = []
        for token in doc:
            if not (token.is_stop or token.is_punct or token.like_num):
                processed_text.append(token.lemma_)
        text = " ".join(processed_text)
        return re.sub(" +", " ", text)