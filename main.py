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
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Falha ao buscar dados da Binance", "status_code": response.status_code}

    candles = response.json()

    if not candles or not isinstance(candles, list):
        return {"error": "Dados inválidos recebidos da Binance"}

    try:
        closes = [float(c[4]) for c in candles]
        last = closes[-1]
        prev = closes[-2]
    except Exception as e:
        return {"error": f"Erro ao processar candles: {str(e)}"}

    # Simular duas classes (para evitar erro do modelo)
    X = [[prev, last], [last, prev]]
    y = [1, 0]

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    prob = model.predict_proba([[last, last]])[0][1]

    return {
        "prob_up": round(prob * 100, 2),
        "prediction": "alta" if prob > 0.5 else "baixa"
    }
