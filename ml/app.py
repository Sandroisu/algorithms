from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -----------------------------
# 1. Создаём приложение FastAPI
# -----------------------------
app = FastAPI(title="House Price Prediction API")

# -----------------------------
# 2. Загружаем обученную модель
# -----------------------------
# Убедись, что в той же папке лежит 'linear_regression_model.pkl'
model = joblib.load("linear_regression_model.pkl")


# -----------------------------
# 3. Определяем структуру входных данных
# -----------------------------
# Pydantic автоматически проверит правильность типов
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# -----------------------------
# 4. Эндпоинт для предсказания
# -----------------------------
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Преобразуем входные данные в numpy-массив
    X = np.array([[features.MedInc,
                   features.HouseAge,
                   features.AveRooms,
                   features.AveBedrms,
                   features.Population,
                   features.AveOccup,
                   features.Latitude,
                   features.Longitude]])

    # Получаем предсказание
    prediction = model.predict(X)[0]

    # Возвращаем результат в JSON-формате
    return {"predicted_price": float(prediction)}


# -----------------------------
# 5. Тестовый эндпоинт
# -----------------------------
@app.get("/")
def root():
    return {"message": "ML модель запущена и готова к предсказаниям!"}
