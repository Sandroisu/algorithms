import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeaturePipeline:
    """
    Класс, инкапсулирующий базовый feature engineering pipeline:
    - нормализация числовых признаков
    - one-hot кодирование категориальных
    - сохранение порядка признаков

    Использует scikit-learn Pipeline и ColumnTransformer под капотом.
    """

    def __init__(self, numeric_features, categorical_features):
        """
        numeric_features: list[str]
            Имена числовых признаков (например: ['age', 'income'])
        categorical_features: list[str]
            Имена категориальных признаков (например: ['gender', 'city'])
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline = None

    def build_pipeline(self):
        """
        Создаёт sklearn-пайплайн с обработчиками для числовых и категориальных колонок.
        """
        # Для числовых признаков — стандартизация (z-score)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Для категориальных признаков — one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Объединяем всё в один ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

        # Сохраняем внутренний пайплайн
        self.pipeline = preprocessor

    def fit(self, df: pd.DataFrame):
        """
        Обучаем трансформеры (например, StandardScaler вычисляет mean/std).
        """
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Применяем все преобразования к новым данным.
        Возвращает numpy-массив (готовый для модели).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built or fitted yet.")
        return self.pipeline.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        fit() + transform() — в одном вызове.
        """
        if self.pipeline is None:
            self.build_pipeline()
        return self.pipeline.fit_transform(df)

    def get_feature_names(self):
        """
        Возвращает список всех имён признаков после трансформации (чтобы понимать, какие колонки куда ушли).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built yet.")
        cat_names = list(self.pipeline.named_transformers_['cat']
                         .named_steps['encoder']
                         .get_feature_names_out(self.categorical_features))
        return self.numeric_features + cat_names
