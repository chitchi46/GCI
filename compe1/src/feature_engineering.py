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
    try:
        # 訓練データに基づいてビンの境界を決定 (ラベルはまだ付けない)
        _, fare_bins_edges = pd.qcut(X_train_fe['Fare'], q=4, retbins=True, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: Could not create 4 quantiles for Fare due to duplicate edges: {e}. Using default quantile-based bins.")
        # デフォルトの分位点を使用するフォールバック
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        fare_bins_edges = X_train_fe['Fare'].quantile(quantiles).to_numpy()
        # 重複する境界値を削除 (ユニーク性を保証)
        fare_bins_edges = np.unique(fare_bins_edges)
        if len(fare_bins_edges) < 2: # ビンが作れない場合
            print("Error: Not enough unique bin edges for Fare. Skipping Fare binning.")
            # Fare_bin を作成せずに元のFareを維持するか、エラーを出すか。ここでは元のFareを維持する想定はしていない。
            # 安全のため、ここでは処理を中断せずに進めるが、実際のプロジェクトではエラーハンドリングを検討。
            # 代わりに、非常に単純なビン１つにするなど。
            fare_bins_edges = [-np.inf, np.inf] # 単一のビン

    # `fare_bins_edges` の最初と最後を調整して、範囲外の値もカバーできるようにする
    if not np.isneginf(fare_bins_edges[0]): # -np.inf でない場合に追加
        fare_bins_edges = np.insert(fare_bins_edges, 0, -np.inf)
    if not np.isposinf(fare_bins_edges[-1]): # np.inf でない場合に追加
        fare_bins_edges = np.append(fare_bins_edges, np.inf)
    
    # 重複した境界値を削除（-inf, inf追加後に発生する可能性もあるため）
    fare_bins_edges = np.unique(fare_bins_edges)

    # fare_bins_edges が確定した後に fare_labels を定義する
    # ビン数が1つしかない場合(例: [-np.inf, np.inf])、ラベルは0のみ
    fare_labels = list(range(max(1, len(fare_bins_edges) - 1))) 
    if len(fare_bins_edges) <=1:
        print("Warning: Fare binning resulted in too few bins. Check Fare distribution or qcut parameters.")
        # この場合、pd.cutはエラーになるので、単一の値を割り当てるなど、別の対応が必要
        # 例: X_train_fe['Fare_bin'] = 0, X_test_fe['Fare_bin'] = 0
        # しかし、その場合はモデルにとって意味のある特徴量にならない可能性が高い。
        # ここでは、後続の pd.cut でエラーが出るのに任せる (より堅牢なエラー処理が必要な場面)

    if len(fare_labels) == 0 and len(fare_bins_edges) == 2: # [-inf, inf]のような場合、ラベルは一つ(0)であるべき
        fare_labels = [0]
    elif len(fare_labels) != len(fare_bins_edges) -1:
        print(f"Error: Mismatch between number of fare labels ({len(fare_labels)}) and bins ({len(fare_bins_edges)-1}). Skipping Fare_bin creation for safety.")
        # 安全のため、Fare_binカラムを作成しない、または元のFareを維持するなどの措置
    else:
        X_train_fe['Fare_bin'] = pd.cut(X_train_fe['Fare'], bins=fare_bins_edges, labels=fare_labels, right=False, include_lowest=True)
        X_test_fe['Fare_bin'] = pd.cut(X_test_fe['Fare'], bins=fare_bins_edges, labels=fare_labels, right=False, include_lowest=True)
        print("Fare binned.")

    # --- 不要になった元のカラムの削除 ---
    columns_to_drop_fe = ['Age', 'Fare']
    # dropする前に、Fare_binが作成されたか確認 (エラーでスキップされた場合を考慮)
    if 'Fare_bin' not in X_train_fe.columns:
        print("Fare_bin was not created, original Fare column will be kept if it exists.")
        if 'Fare' in columns_to_drop_fe: columns_to_drop_fe.remove('Fare')
            
    X_train_fe.drop(columns_to_drop_fe, axis=1, inplace=True, errors='ignore') # errors='ignore'で存在しないカラムを無視
    X_test_fe.drop(columns_to_drop_fe, axis=1, inplace=True, errors='ignore')
    print(f"Dropped specified original columns: {columns_to_drop_fe}")

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