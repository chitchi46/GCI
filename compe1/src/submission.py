# 提出ファイルの作成を担当するモジュール
import pandas as pd

SUBMISSION_DIR = "../results"
SAMPLE_SUBMISSION_PATH = "../data/sample_submission.csv"

def create_submission_file(test_df_with_id: pd.DataFrame, predictions: list, experiment_id: str, target_col: str = "Perished"):
    """提出ファイルを作成する関数"""
    # TODO: 実装
    pass 