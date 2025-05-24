# モデルの定義、学習、評価を担当するモジュール
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pickle
import os

MODEL_DIR = "../models"

def train_lgbm_cv(X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int = 5, random_state: int = 42) -> tuple:
    """LightGBMモデルをクロスバリデーションで学習・評価する関数"""
    # TODO: 実装
    pass

def save_model(model, model_name: str):
    """学習済みモデルを保存する関数"""
    # TODO: 実装
    pass

def load_model(model_name: str):
    """学習済みモデルを読み込む関数"""
    # TODO: 実装
    pass 