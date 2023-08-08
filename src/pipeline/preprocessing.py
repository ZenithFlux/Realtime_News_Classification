import re

import numpy as np
import spacy

from ..logger import logging as log

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray


class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.preprocess = np.vectorize(self._preprocess)
        
    def process_data(self, X: 'ndarray | list[str]'):
        X = np.array(X, dtype=object)
        X = self.preprocess(X)
        log.info("Data Transformation completed...")
        return X
        
    def _preprocess(self, text: str):
        doc = self.nlp(text)
        processed_text = []
        for token in doc:
            if not (token.is_stop or token.is_punct or token.like_num):
                processed_text.append(token.lemma_)
        text = " ".join(processed_text)
        return re.sub(" +", " ", text)