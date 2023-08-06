from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib as jl

from ..logger import logging as log

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .data import ClassifierDataset
    from pathlib import Path


class NewsClassifier:
    def __init__(self, load_path: 'Path | str | None' = None):
        if load_path is not None:
            self = jl.load(str(load_path))
        else :
            self.vectorizer = TfidfVectorizer()
            self.model = LogisticRegression()
            self.label_map = LabelEncoder()
        
    def fit(self, data: 'ClassifierDataset'):
        X = self.vectorizer.fit_transform(data.X_train)
        Y = self.label_map.fit_transform(data.Y_train)
        self.model.fit(X, Y)
        log.info("Model Training complete...")
        
    def test_pred(self, data):
        X = self.vectorizer.transform(data)
        return self.model.predict(X)
        
    def predict(self, data):
        pred = self.test_pred(data)
        return self.label_map.inverse_transform(pred)
    
    def save(self, path: 'Path | str'="."):
        jl.dump(self, str(path))