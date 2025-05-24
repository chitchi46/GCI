# 特徴量エンジニアリングを担当するモジュール
import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder # 今回は直接使わないが、将来的に使う可能性あり

def create_base_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基本的な特徴量エンジニアリングを行う。
    AgeとFareのビン化など。

    Args:
        X_train (pd.DataFrame): 前処理済みの訓練データの特徴量
        X_test (pd.DataFrame): 前処理済みのテストデータの特徴量

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 特徴量エンジニアリング後の訓練データとテストデータ
    """
    print("Starting feature engineering (base features)...")
    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    # --- Age のビン化 ---
    age_bins = [0, 12, 18, 25, 35, 50, 65, 81] # 0-12, 13-18, 19-25, 26-35, 36-50, 51-65, 66-80
    age_labels = list(range(len(age_bins) - 1)) # [0, 1, 2, 3, 4, 5, 6]
    
    X_train_fe['Age_bin'] = pd.cut(X_train_fe['Age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
    X_test_fe['Age_bin'] = pd.cut(X_test_fe['Age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
    
    # pd.cut は Categorical型を返すので、数値として扱いたい場合は .astype(int) などが必要になる場合がある。
    # LightGBMはCategorical型を直接扱えるので、そのままでも良い。今回は数値ラベルなので問題ないはず。
    # 欠損値がもしあれば補完が必要だが、Ageはpreprocessorで補完済み想定。
    print("Age binned.")

    # --- Fare のビン化 (対数変換後のFareに対して) ---
    # 訓練データに基づいてビンの境界を決定
    # duplicates='raise' (デフォルト) にして、境界値が一意でない場合にエラーを出すようにし、ビンの数を調整するアプローチも検討できる
    try:
        X_train_fe['Fare_bin'], fare_bins_edges = pd.qcut(X_train_fe['Fare'], q=4, retbins=True, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: Could not create 4 quantiles for Fare due to duplicate edges: {e}. Trying with fewer bins or different strategy.")
        # 代替戦略: 例えば3分位にするか、固定の境界値を設定する
        # ここでは簡略化のため、エラー時は処理をスキップせずに続行するが、実際はより堅牢な処理が必要
        # もし_fare_bins_edgesが未定義で終わるのを避けるため、デフォルト値を設定
        fare_bins_edges = [-np.inf, X_train_fe['Fare'].quantile(0.25), X_train_fe['Fare'].median(), X_train_fe['Fare'].quantile(0.75), np.inf]

    fare_labels = list(range(len(fare_bins_edges) - 1))

    # `fare_bins_edges` の最初と最後を調整して、範囲外の値もカバーできるようにする
    # pd.qcut(duplicates='drop') を使った場合、fare_bins_edges の数は q+1 より少なくなることがあるので、ラベル数もそれに合わせる必要がある。
    # retbins=Trueで得られた境界値を使うので、 labels の数は len(fare_bins_edges) - 1 となる。
    if fare_bins_edges[0] != -np.inf:
        fare_bins_edges = np.insert(fare_bins_edges, 0, -np.inf)
    if fare_bins_edges[-1] != np.inf:
        fare_bins_edges = np.append(fare_bins_edges, np.inf)
    
    # X_train_fe['Fare_bin'] は既にqcutで作成済みなので、X_test_feのみpd.cutで作成
    X_test_fe['Fare_bin'] = pd.cut(X_test_fe['Fare'], bins=fare_bins_edges, labels=fare_labels, right=False, include_lowest=True)

    # Fareの欠損値はpreprocessorで補完済み想定。
    print("Fare binned.")

    # --- 不要になった元のカラムの削除 ---
    columns_to_drop_fe = ['Age', 'Fare']
    X_train_fe.drop(columns_to_drop_fe, axis=1, inplace=True)
    X_test_fe.drop(columns_to_drop_fe, axis=1, inplace=True)
    print(f"Dropped original columns: {columns_to_drop_fe}")

    print(f"Features after base FE: {X_train_fe.columns.tolist()}")
    print("Feature engineering (base features) finished.")
    return X_train_fe, X_test_fe

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を作成する関数"""
    # TODO: 実装
    pass

def select_features(df: pd.DataFrame, target_col: str) -> list:
    """特徴量選択を行う関数"""
    # TODO: 実装
    pass 