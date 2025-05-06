from fastapi import FastAPI
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "API do CandleAI ativa 🔥"}

@app.get("/predict")
def predict():
    # 1. Coleta dados da Binance
    response = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60")
    candles = response.json()
    
    closes = [float(c[4]) for c in candles]
    last = closes[-1]
    prev = closes[-2]
    
    # 2. Exemplo com 2 classes
    X = [[prev, last], [last, prev]]
    y = [1, 0]

    model = RandomForestClassifier()
    model.fit(X, y)
    prob = model.predict_proba([[last, last]])[0][1]

    return {
        "prob_up": round(prob * 100, 2),
        "prediction": "alta" if prob > 0.5 else "baixa"
    }
