# -*- coding: utf-8 -*-
import pandas as pd
from src import config # config.py から定数をインポート

def verify_data_integrity(df, df_name="DataFrame"):
    """
    データフレームの基本的な整合性を確認し、情報を表示する関数
    """
    print(f"--- Data Integrity Check for {df_name} ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values (Count):")
    missing_values_count = df.isnull().sum()
    print(missing_values_count[missing_values_count > 0])
    print("\nMissing Values (Percentage):")
    missing_values_percent = (df.isnull().sum() / len(df)) * 100
    print(missing_values_percent[missing_values_percent > 0].sort_values(ascending=False))

    if df_name == "train_df": # 学習データの場合のみ重複をチェック
        duplicate_rows = df.duplicated().sum()
        print(f"\nDuplicate Rows: {duplicate_rows}")
        # ターゲットのクラスバランスを表示
        if config.TARGET_COLUMN in df.columns:
            print(f"\nTarget Column ({config.TARGET_COLUMN}) Distribution:")
            print(df[config.TARGET_COLUMN].value_counts(normalize=True) * 100)
        else:
            print(f"\nTarget column '{config.TARGET_COLUMN}' not found in {df_name}.")

    print(f"--- End of Integrity Check for {df_name} ---\n")

def load_data(): # 引数を削除し、configからパスを取得
    """
    学習データとテストデータを読み込む関数
    """
    print(f"Loading train data from: {config.TRAIN_DATA_PATH}")
    print(f"Loading test data from: {config.TEST_DATA_PATH}")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    verify_data_integrity(train_df, "train_df")
    verify_data_integrity(test_df, "test_df")

    return train_df, test_df 