# SMS Spam Classifier — FastAPI Website

This project converts the uploaded `sms_spam_classifier.ipynb` ML project into a simple FastAPI + HTML/CSS website.

## ML pipeline used

The web app follows the important choices in the notebook:

- Lowercase text
- NLTK tokenization
- Remove non-alphanumeric tokens
- Remove English stopwords
- Porter stemming
- TF-IDF vectorization with `max_features=3000`
- Multinomial Naive Bayes classifier
- `train_test_split(test_size=0.2, random_state=2)`

## Project structure

```text
sms_spam_classifier_fastapi/
├── main.py
├── train_model.py
├── requirements.txt
├── README.md
├── sms_spam_classifier.ipynb
├── spam.csv                 # ADD YOUR DATASET HERE
├── model/
│   ├── model.pkl            # generated after training
│   └── vectorizer.pkl       # generated after training
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup on Windows

Open a terminal in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put the same `spam.csv` dataset used by the notebook in the project root.

Then train/save the model:

```bash
python train_model.py
```

Start the website:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Important

The uploaded notebook refers to `spam.csv`, but that CSV was not included in the uploaded file. Therefore this ZIP does not fabricate or replace your training dataset. Add your original `spam.csv` before running `train_model.py`.

The generated `model/model.pkl` and `model/vectorizer.pkl` are intentionally created locally from your dataset rather than shipping a different pre-trained model.

## API

Health check:

```text
GET /health
```

The main prediction endpoint is:

```text
POST /predict
```

with form field:

```text
message
```
