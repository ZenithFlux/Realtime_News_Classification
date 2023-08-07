from src.infer import load_model
import numpy as np

model = load_model()

pred = model.predict(["The criminal is sentenced to jail", "j"])
print(pred)