from src.infer import load_model
from flask import Flask, request


model = load_model()
application = app = Flask(__name__)

@app.route("/", methods=["POST"])
def get_news_type():
    "Receives data from 'article' field of form data"
    
    pred = model.predict([request.form.get("article")])
    return {"news_type": pred[0]}