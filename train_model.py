import os
import pickle
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()


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


def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "spam.csv was not found. Put the SMS Spam Collection CSV file "
            "next to train_model.py and run this script again."
        )

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Same cleaning used in the uploaded notebook.
    drop_cols = [c for c in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    if "v1" in df.columns and "v2" in df.columns:
        df.rename(columns={"v1": "Target", "v2": "Text"}, inplace=True)

    if "Target" not in df.columns or "Text" not in df.columns:
        raise ValueError(
            "CSV must contain v1/v2 columns (or Target/Text columns), "
            "matching the uploaded notebook."
        )

    df["Target"] = df["Target"].map({"ham": 0, "spam": 1})
    df.dropna(subset=["Target", "Text"], inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df["Target"] = df["Target"].astype(int)

    df["Transformed_text"] = df["Text"].apply(transform_text)

    # Same vectorizer/model choice as the uploaded notebook.
    tfid = TfidfVectorizer(max_features=3000)
    X = tfid.fit_transform(df["Transformed_text"]).toarray()
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfid, f)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")


if __name__ == "__main__":
    train()
