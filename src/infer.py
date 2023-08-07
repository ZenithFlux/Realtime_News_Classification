from .pipeline.model import NewsClassifier
from .paths import MODEL_PATH

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path

def load_model(model_path: 'Path | str'=MODEL_PATH):
    return NewsClassifier(model_path)