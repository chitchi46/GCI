# Optuna を用いたハイパーパラメータチューニングを担当するモジュール
import optuna
import pandas as pd
from .model import train_lgbm_cv # model.py から関数をインポート

def objective_lgbm(trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> float:
    """LightGBM の Optuna 最適化関数"""
    # TODO: 実装
    pass

def run_optuna_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 100, n_splits: int = 5, random_state: int = 42) -> optuna.study.Study:
    """LightGBM の Optuna チューニングを実行する関数"""
    # TODO: 実装
    pass 