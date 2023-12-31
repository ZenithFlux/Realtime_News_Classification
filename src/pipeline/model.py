import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib as jl

from ..logger import logging as log
from .preprocessing import DataPreprocessor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from numpy import ndarray


class NewsClassifier:
    def __init__(self, load_path: 'Path | str | None' = None):
        if load_path is not None:
            m = jl.load(str(load_path))
            
            # updating attributes of current model with loaded model
            self.__dict__.update(m.__dict__)
            
        else :
            self.vectorizer = TfidfVectorizer()
            self.model = LogisticRegression()
            self.le = LabelEncoder()
            
        self.pp = DataPreprocessor()
        
    def fit(self, X: 'ndarray | list[str]', Y: 'ndarray | list[str]'):
        "Fit the model to given data"
        
        X = self.pp.process_data(X)
        
        X = self.vectorizer.fit_transform(X)
        Y = self.le.fit_transform(Y)
        self.model.fit(X, Y)
        log.info("Model Training complete...")
    
    def evaluate(self, X: 'ndarray | list[str]', Y: 'ndarray | list[str]') -> dict[str, float]:
        """
        Evalute model using given data
        
        Returns a dict with scores of various evaluation metrics.
        Metrics used: accuracy, precision, recall, f1_score, roc_auc.
        """
        
        X = self.pp.process_data(X)
        
        X = self.vectorizer.transform(X)
        Y = self.le.transform(Y)
        
        y = self.model.predict(X)
        scores = {"accuracy": accuracy_score(Y, y),
                "precision": precision_score(Y, y, average="macro"),
                "recall": recall_score(Y, y, average="macro"),
                "f1_score": f1_score(Y, y, average="macro"),
                "roc_auc": roc_auc_score(Y, self.model.predict_proba(X), 
                                         average="macro", multi_class="ovo")}
        
        log.info("Model Evaluation complete...")
        return scores
    
    def test_pred(self, data: 'ndarray | list[str]') -> 'ndarray':
        "Predicts class indices"
        
        X = self.pp.process_data(data)
        X = self.vectorizer.transform(data)
        return self.model.predict(X)
        
    def predict(self, data: 'ndarray | list[str]') -> 'ndarray':
        "Predicts class labels"
        
        pred = self.test_pred(data)
        return self.le.inverse_transform(pred)
    
    def save(self, path: 'Path | str'="."):
        "Saves the model for future use"
        
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.pp = None
        jl.dump(self, path)
        self.pp = DataPreprocessor()