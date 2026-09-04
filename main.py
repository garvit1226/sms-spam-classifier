import os
import pickle
import string

import nltk
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

app = FastAPI(title="SMS Spam Classifier")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def transform_text(text: str) -> str:
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)

    cleaned = []
    for token in tokens:
        if token.isalnum():
            cleaned.append(token)

    cleaned = [
        token for token in cleaned
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(ps.stem(token) for token in cleaned)


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "message": "",
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    global model, vectorizer

    message = message.strip()

    if not message:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": None,
                "message": "",
                "error": None,
            },
        )

    if model is None or vectorizer is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": None,
                "message": message,
                "error": (
                    "Model files are missing. Add spam.csv and run "
                    "'python train_model.py' first."
                ),
            },
        )

    transformed = transform_text(message)
    vector = vectorizer.transform([transformed])

    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities) * 100)

    label = "SPAM" if prediction == 1 else "REAL"
    css_class = "spam" if prediction == 1 else "real"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": label,
            "css_class": css_class,
            "confidence": round(confidence, 2),
            "message": message,
            "error": None,
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and vectorizer is not None,
    }
