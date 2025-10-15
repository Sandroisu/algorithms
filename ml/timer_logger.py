import logging
import time
from contextlib import contextmanager

# -----------------------------
# 1. базовая настройка логгера
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------
# 2. контекстный менеджер для замера времени блока
# -----------------------------
@contextmanager
def timer(name: str):
    """
    Используется как:
        with timer("обучение модели"):
            model.fit(X, y)
    После выхода из блока выведет в лог время выполнения.
    """
    t0 = time.time()
    logging.info(f"[{name}] start ...")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        logging.info(f"[{name}] done in {elapsed:.3f} sec")

# -----------------------------
# 3. пример использования
# -----------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(2000, 10)
    y = np.random.randint(0, 2, size=2000)

    with timer("fit RandomForest"):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

    with timer("prediction"):
        preds = model.predict(X)
        acc = (preds == y).mean()
        logging.info(f"train accuracy = {acc:.3f}")
