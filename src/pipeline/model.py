import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib as jl

from ..logger import logging as log

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path


class NewsClassifier:
    def __init__(self, load_path: 'Path | str | None' = None):
        if load_path is not None:
            m = jl.load(str(load_path))
            self.__dict__.update(m.__dict__)
        else :
            self.vectorizer = TfidfVectorizer()
            self.model = LogisticRegression()
            self.le = LabelEncoder()
        
    def fit(self, X, Y):
        X = self.vectorizer.fit_transform(X)
        Y = self.le.fit_transform(Y)
        self.model.fit(X, Y)
        log.info("Model Training complete...")
    
    def evaluate(self, X, Y) -> dict[str, float]:
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
    
    def test_pred(self, data):
        X = self.vectorizer.transform(data)
        return self.model.predict(X)
        
    def predict(self, data):
        pred = self.test_pred(data)
        return self.le.inverse_transform(pred)
    
    def save(self, path: 'Path | str'="."):
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        jl.dump(self, path)