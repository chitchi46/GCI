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

def load_train_data(file_path: str) -> pd.DataFrame:
    """訓練データを読み込む関数"""
    df = pd.read_csv(file_path)
    print(f"{file_path} を読み込みました。Shape: {df.shape}")
    return df

def load_test_data(file_path: str) -> pd.DataFrame:
    """テストデータを読み込む関数"""
    df = pd.read_csv(file_path)
    print(f"{file_path} を読み込みました。Shape: {df.shape}")
    return df

def check_data_integrity(df: pd.DataFrame, df_name: str):
    """データの整合性をチェックする関数 (null, dtype, duplicates)"""
    print(f"\n--- {df_name} の整合性チェック ---")
    print("\n[基本情報]")
    print(f"Shape: {df.shape}")

    print("\n[欠損値の割合 (%)]")
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    null_info = pd.DataFrame({'欠損数': null_counts, '割合 (%)': null_percentages})
    print(null_info[null_info['欠損数'] > 0].sort_values(by='割合 (%)', ascending=False))

    print("\n[データ型]")
    print(df.dtypes)

    print("\n[重複行の数]")
    duplicate_rows = df.duplicated().sum()
    print(f"{duplicate_rows} 行")

    if duplicate_rows > 0:
        print("重複行を削除します。")
        df.drop_duplicates(inplace=True)
        print(f"削除後のShape: {df.shape}")
    print(f"--- {df_name} の整合性チェック完了 ---")
    return df # 重複削除後のDataFrameを返す場合があるため 