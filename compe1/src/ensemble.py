# モデルのアンサンブリングを担当するモジュール
import pandas as pd
import numpy as np

def average_predictions(predictions: list) -> np.ndarray:
    """複数の予測結果を平均する関数 (ソフトボーティング)"""
    # TODO: 実装
    pass

def majority_vote_predictions(predictions: list) -> np.ndarray:
    """複数の予測結果で多数決を取る関数 (ハードボーティング)"""
    # TODO: 実装
    pass 