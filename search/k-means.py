from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

class KMeans:
    """
    Минималистичная реализация k-means с нуля (без scikit-learn).

    Алгоритм:
      1) Инициализируем K центроидов (random или k-means++).
      2) Повторяем до сходимости или до max_iter:
         a) E-шаг: назначаем каждую точку ближайшему центроиду (по L2).
         b) M-шаг: пересчитываем центроиды как средние своих кластеров.
      3) Критерий оптимизации (инерция) — сумма квадратов расстояний точек до центроидов.

    Параметры
    ---------
    n_clusters : int
        Число кластеров K.
    init : str
        'random' или 'k-means++' — способ инициализации центроидов.
    max_iter : int
        Максимум итераций EM-процесса.
    tol : float
        Порог сходимости: если сдвиг центроидов между итерациями меньше tol, останавливаемся.
    random_state : Optional[int]
        Фиксируем seed генератора случайных чисел для воспроизводимости.
    n_init : int
        Сколько раз перезапустить k-means с разной инициализацией и взять лучшее по инерции.

    Атрибуты (после fit)
    --------------------
    cluster_centers_ : np.ndarray, shape (K, D)
        Итоговые центроиды.
    labels_ : np.ndarray, shape (N,)
        Метка кластера для каждой точки обучающего набора.
    inertia_ : float
        Сумма квадратов расстояний (чем меньше, тем лучше для данного K).
    n_iter_ : int
        Сколько итераций потребовалось в лучшем запуске.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        n_init: int = 10,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if init not in {"random", "k-means++"}:
            raise ValueError("init must be 'random' or 'k-means++'")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if n_init <= 0:
            raise ValueError("n_init must be positive")

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        # Будут заполнены после fit()
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    # -------------------------------
    # Вспомогательные функции
    # -------------------------------

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        """Проверяем входные данные: приводим к float64 и убеждаемся в 2D форме."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features)")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        return X

    def _euclidean_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу расстояний L2 от каждой точки до каждого центроида.
        Векторизация: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        Возвращает массив shape (N, K).
        """
        X_sq = np.sum(X**2, axis=1, keepdims=True)          # (N, 1)
        C_sq = np.sum(centers**2, axis=1, keepdims=True).T  # (1, K)
        # Матрица скалярных произведений X @ centers^T имеет shape (N, K)
        # Итог: расстояния^2
        dists_sq = X_sq + C_sq - 2.0 * X @ centers.T
        # Из-за численных ошибок может появиться -0.0 — обрезаем в ноль
        np.maximum(dists_sq, 0, out=dists_sq)
        return np.sqrt(dists_sq, dtype=np.float64)

    def _init_centers_random(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Случайно выбираем K различных точек датасета как начальные центроиды."""
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices]

    def _init_centers_kmeanspp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Инициализация k-means++:
        1) Случайно выбираем первый центроид.
        2) Каждый следующий выбираем пропорционально D(x)^2 — расстоянию до ближайшего уже выбранного центроида.
        Это снижает риск «плохого старта» и ускоряет сходимость.
        """
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)

        # 1) Первый центроид — случайная точка
        first = rng.integers(n_samples)
        centers[0] = X[first]

        # 2) Поочередно добавляем центроиды
        # D(x): расстояние от точки до ближайшего уже выбранного центра
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for i in range(1, self.n_clusters):
            # Вероятности пропорциональны D(x)^2
            probs = closest_dist_sq / closest_dist_sq.sum()
            # Сэмплируем индекс новой точки-центра
            next_idx = rng.choice(n_samples, p=probs)
            centers[i] = X[next_idx]
            # Обновляем D(x) с учетом нового центра
            new_dist_sq = np.sum((X - centers[i]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Выбор стратегии инициализации центров."""
        if self.init == "random":
            return self._init_centers_random(X, rng)
        else:
            return self._init_centers_kmeanspp(X, rng)

    def _compute_inertia(
        self, X: np.ndarray, centers: np.ndarray, labels: np.ndarray
    ) -> float:
        """Сумма квадратов расстояний точек до назначенного центроида."""
        diffs = X - centers[labels]
        return float(np.sum(diffs * diffs))

    # -------------------------------
    # Публичный API
    # -------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Обучаем k-means: находим центры и назначения точек.
        Запускаем n_init раз с разной инициализацией и берём лучший по инерции.
        """
        X = self._check_X(X)
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = None

        # Генератор случайных чисел
        rng_master = np.random.default_rng(self.random_state)

        for init_run in range(self.n_init):
            # Для каждого запуска деривируем дочерний RNG, чтобы старты отличались, но были воспроизводимы
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            centers = self._init_centers(X, rng)

            for it in range(1, self.max_iter + 1):
                # E-шаг: назначаем кажду точку ближайшему центру
                dists = self._euclidean_distances(X, centers)  # (N, K)
                labels = np.argmin(dists, axis=1)               # (N,)

                # M-шаг: пересчёт центроидов как средних своих кластеров
                new_centers = np.empty_like(centers)
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if np.any(mask):
                        new_centers[k] = X[mask].mean(axis=0)
                    else:
                        # Пустой кластер: «реинициализируем» центр случайной точкой,
                        # чтобы алгоритм мог продолжиться.
                        new_centers[k] = X[rng.integers(X.shape[0])]

                # Проверка сходимости: насколько сильно сдвинулись центры
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift <= self.tol:
                    break

            inertia = self._compute_inertia(X, centers, labels)

            # Сохраняем лучший запуск
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_n_iter = it

        # Итог
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Назначаем новые точки ближайшему кластеру.
        Важно: центры берём из уже обученной модели (fit должен быть вызван).
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit(X) before predict(X).")
        X = self._check_X(X)
        dists = self._euclidean_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Удобный шорткат: обучаемся и сразу возвращаем labels."""
        self.fit(X)
        return self.labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращаем расстояния до центроидов (полезно для фич-инжиниринга:
        можно использовать расстояния как новые признаки).
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit(X) before transform(X).")
        X = self._check_X(X)
        return self._euclidean_distances(X, self.cluster_centers_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit + transform в одном вызове (как в sklearn)."""
        self.fit(X)
        return self.transform(X)
