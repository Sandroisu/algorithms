# Импортируем нужные библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # для сохранения модели

# -----------------------------
# 1. Загружаем данные
# -----------------------------
# Возьмём встроенный датасет из sklearn (например, цены домов в Бостоне)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame  # это DataFrame, как таблица в Excel

print("Форма данных:", df.shape)
print(df.head())

# -----------------------------
# 2. Разделяем на фичи (X) и целевую переменную (y)
# -----------------------------
X = df.drop(columns=['MedHouseVal'])  # признаки (доход, возраст дома, население и т.д.)
y = df['MedHouseVal']                 # целевая переменная (цена дома)

# -----------------------------
# 3. Разделяем на train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Обучаем модель
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Делаем предсказание и оцениваем ошибку
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Среднеквадратичная ошибка (MSE):", mse)

# -----------------------------
# 6. Сохраняем модель
# -----------------------------
joblib.dump(model, "linear_regression_model.pkl")
print("Модель сохранена в файл 'linear_regression_model.pkl'")

# -----------------------------
# 7. Загружаем модель обратно (проверим)
# -----------------------------
loaded_model = joblib.load("linear_regression_model.pkl")
print("Загруженная модель делает то же самое:", loaded_model.predict(X_test[:1]))
