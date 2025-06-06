# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src import config # config.py から定数をインポート

def analyze_missing_patterns(df: pd.DataFrame, df_name: str):
    """欠損値のパターンを詳細に分析"""
    print(f"\n=== Missing Value Analysis for {df_name} ===")
    
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Missing Value Summary:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # 欠損値の組み合わせパターンを分析
    if missing_count.sum() > 0:
        print("\nMissing Value Patterns:")
        missing_patterns = df.isnull().value_counts()
        print(missing_patterns.head(10))
    
    return missing_df

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr'):
    """外れ値を検出"""
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return pd.Series(False, index=df.index)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        threshold = 3
        outliers = pd.Series(False, index=df.index)
        outliers.loc[df[column].notna()] = z_scores > threshold
    
    else:
        outliers = pd.Series(False, index=df.index)
    
    return outliers

def advanced_age_imputation(combined_df: pd.DataFrame):
    """より高度なAge補完（設定に基づく）"""
    method = config.AGE_IMPUTATION_METHOD
    print(f"Advanced Age imputation using {method}...")
    
    # Age補完用の特徴量準備
    age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                   'Title_Encoded_For_Age', 'FamilySize']
    
    # Deckが使用される場合のみ追加
    if config.CABIN_EXTRACT_DECK:
        age_features.append('Deck_Encoded_For_Age')
    
    # 利用可能な特徴量のみを使用
    available_features = [f for f in age_features if f in combined_df.columns]
    
    age_df = combined_df[['Age'] + available_features].copy()
    
    # Age補完用データセットの準備
    age_known = age_df[age_df['Age'].notnull()].copy()
    age_unknown = age_df[age_df['Age'].isnull()].copy()
    
    if not age_unknown.empty and len(age_known) > 10:  # 最低限のデータ数を確保
        # 欠損値を一時的に補完（Age以外の特徴量）
        for col in available_features:
            if age_known[col].isnull().any():
                if combined_df[col].dtype in ['int64', 'float64']:
                    fill_value = age_known[col].median()
                else:
                    fill_value = age_known[col].mode()[0] if not age_known[col].mode().empty else 0
                
                age_known[col] = age_known[col].fillna(fill_value)
                age_unknown[col] = age_unknown[col].fillna(fill_value)
        
        X_age_train = age_known[available_features]
        y_age_train = age_known['Age']
        X_age_predict = age_unknown[available_features]
        
        # 手法に応じてモデルを選択
        if method == "random_forest":
            model = RandomForestRegressor(
                n_estimators=config.AGE_IMPUTATION_PARAMS['n_estimators'],
                max_depth=config.AGE_IMPUTATION_PARAMS['max_depth'],
                random_state=config.AGE_IMPUTATION_PARAMS['random_state']
            )
        elif method == "linear_regression":
            model = LinearRegression()
        else:
            # デフォルトはRandomForest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=config.RANDOM_STATE
            )
        
        # モデル学習と予測
        model.fit(X_age_train, y_age_train)
        predicted_age = model.predict(X_age_predict)
        
        # 合理的な範囲にクリップ
        predicted_age = np.clip(predicted_age, 
                              config.AGE_CLIP_RANGE[0], 
                              config.AGE_CLIP_RANGE[1])
        
        # 元のデータフレームに代入
        combined_df.loc[combined_df['Age'].isnull(), 'Age'] = predicted_age
        
        print(f"Age imputation completed. Imputed {len(age_unknown)} missing values.")
        
        # RandomForestの場合は特徴量重要度を表示
        if method == "random_forest":
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Age imputation feature importance:")
            print(feature_importance)
            
    else:
        # データが不十分な場合は中央値で補完
        print("Insufficient data for advanced imputation. Using median.")
        combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())
    
    return combined_df

def create_statistical_features(combined_df: pd.DataFrame):
    """統計的特徴量の作成"""
    print("Creating statistical features...")
    
    # 年齢グループ
    combined_df['AgeGroup'] = pd.cut(combined_df['Age'], 
                                   bins=config.AGE_BINS, 
                                   labels=config.AGE_LABELS)
    
    # 運賃グループ（対数変換前の値を使用）
    fare_original = np.expm1(combined_df['Fare'])  # log1p の逆変換
    combined_df['FareGroup'] = pd.qcut(fare_original, q=config.FARE_QUANTILES, 
                                     labels=config.FARE_GROUP_LABELS)
    
    # 家族サイズグループ
    def categorize_family_size(size):
        if size == config.FAMILY_SIZE_MAPPING['single']:
            return 'Single'
        elif size <= config.FAMILY_SIZE_MAPPING['small_max']:
            return 'Small'
        else:
            return 'Large'
    
    combined_df['FamilySizeGroup'] = combined_df['FamilySize'].apply(categorize_family_size)
    
    # タイトルグループ
    combined_df['TitleGroup'] = combined_df['Title'].apply(
        lambda x: 'Common' if x in config.COMMON_TITLES else 'Rare'
    )
    
    # 相互作用特徴量の作成（設定で制御）
    if config.CREATE_INTERACTION_FEATURES:
        # 年齢と運賃の相互作用
        combined_df['Age_Fare_Interaction'] = combined_df['Age'] * combined_df['Fare']
        
        # クラスと運賃の相互作用
        combined_df['Pclass_Fare_Interaction'] = combined_df['Pclass'] * combined_df['Fare']
        
        # 性別と年齢の相互作用
        combined_df['Sex_Age_Interaction'] = combined_df['Sex'] * combined_df['Age']
        
        # クラスと家族サイズの相互作用
        combined_df['Pclass_FamilySize_Interaction'] = combined_df['Pclass'] * combined_df['FamilySize']
        
        print("Interaction features created.")
    
    print("Statistical features created.")
    return combined_df

def handle_outliers(combined_df: pd.DataFrame, columns: list = None, method: str = None):
    """外れ値の処理（設定に基づく）"""
    print("Handling outliers...")
    
    # デフォルト値を設定から取得
    if columns is None:
        columns = config.OUTLIER_COLUMNS
    if method is None:
        method = config.OUTLIER_HANDLING_METHOD
    
    detection_method = config.OUTLIER_DETECTION_METHOD
    
    for col in columns:
        if col in combined_df.columns and combined_df[col].dtype in ['int64', 'float64']:
            outliers = detect_outliers(combined_df, col, method=detection_method)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"Found {outlier_count} outliers in {col} using {detection_method} method")
                
                if method == 'cap':
                    # IQRベースでキャップ
                    Q1 = combined_df[col].quantile(0.25)
                    Q3 = combined_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    combined_df[col] = combined_df[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                elif method == 'winsorize':
                    # 上下1%をカット
                    lower_percentile = combined_df[col].quantile(0.01)
                    upper_percentile = combined_df[col].quantile(0.99)
                    combined_df[col] = combined_df[col].clip(lower=lower_percentile, upper=upper_percentile)
                    print(f"Winsorized {col} to [{lower_percentile:.2f}, {upper_percentile:.2f}]")
    
    return combined_df

def validate_data_quality(combined_df: pd.DataFrame):
    """データ品質の検証"""
    print("\n=== Data Quality Validation ===")
    
    # 基本的な統計情報
    print("Basic Statistics:")
    print(combined_df.describe())
    
    # データ型の確認
    print("\nData Types:")
    print(combined_df.dtypes)
    
    # 重複行の確認
    duplicates = combined_df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # 各カラムの一意値数
    print("\nUnique values per column:")
    for col in combined_df.columns:
        unique_count = combined_df[col].nunique()
        print(f"{col}: {unique_count}")
    
    # 無限値やNaNの確認
    inf_count = np.isinf(combined_df.select_dtypes(include=[np.number])).sum().sum()
    nan_count = combined_df.isnull().sum().sum()
    print(f"\nInfinite values: {inf_count}")
    print(f"NaN values: {nan_count}")
    
    return True

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    改善された前処理を行う関数
    - より詳細な欠損値分析
    - 外れ値の検出と処理
    - 高度なAge補完（RandomForest使用）
    - 統計的特徴量の作成
    - データ品質の検証
    """
    print("=== Starting Enhanced Preprocessing ===")
    
    # ターゲットカラムを学習データから一時的に分離
    y_train_original = train_df[config.TARGET_COLUMN]
    train_df_no_target = train_df.drop(config.TARGET_COLUMN, axis=1)

    # 欠損値パターンの分析
    analyze_missing_patterns(train_df_no_target, "Training Data")
    analyze_missing_patterns(test_df, "Test Data")

    # データ結合
    combined_df = pd.concat([train_df_no_target, test_df], keys=['train', 'test'])
    original_train_index = train_df.index
    original_test_index = test_df.index

    print(f"\nCombined dataset shape: {combined_df.shape}")

    # 1. Name から Title を抽出 & マッピング
    print("\n--- Step 1: Title Extraction ---")
    combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # 設定ファイルからタイトルマッピングを使用
    combined_df['Title'] = combined_df['Title'].map(config.TITLE_MAPPING).fillna('Unknown')
    
    print("Title distribution:")
    print(combined_df['Title'].value_counts())

    # 2. Cabin から Deck を抽出と詳細分析
    print("\n--- Step 2: Cabin/Deck Processing ---")
    if config.CABIN_EXTRACT_DECK:
        combined_df['Deck'] = combined_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
        combined_df['CabinKnown'] = combined_df['Cabin'].notnull().astype(int)
    
    # キャビン番号から部屋番号を抽出（数値部分）
    if config.CABIN_EXTRACT_NUMBER:
        combined_df['CabinNumber'] = combined_df['Cabin'].str.extract(r'(\d+)').astype(float)
    
    print("Deck distribution:")
    print(combined_df['Deck'].value_counts())

    # 3. 基本的なカテゴリ変数のエンコーディング
    print("\n--- Step 3: Basic Categorical Encoding ---")
    
    # Sex のエンコード
    combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Embarked の欠損値を設定に基づいて補完 & エンコード
    if config.EMBARKED_FILL_STRATEGY == "mode":
        mode_embarked = combined_df['Embarked'].mode()[0]
        combined_df['Embarked'] = combined_df['Embarked'].fillna(mode_embarked)
        print(f"Embarked missing values filled with mode: {mode_embarked}")
    else:
        # 他の戦略があれば追加
        combined_df['Embarked'] = combined_df['Embarked'].fillna('S')  # デフォルト
        print("Embarked missing values filled with default: S")
    
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    combined_df['Embarked'] = combined_df['Embarked'].map(embarked_mapping)

    # 4. 家族サイズ関連の特徴量を先に作成（Age補完で使用するため）
    print("\n--- Step 4: Family Features ---")
    combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
    combined_df['IsAlone'] = (combined_df['FamilySize'] == 1).astype(int)
    
    # Title と Deck の一時的なエンコード（Age補完用）
    title_le_for_age = LabelEncoder()
    combined_df['Title_Encoded_For_Age'] = title_le_for_age.fit_transform(combined_df['Title'])

    if config.CABIN_EXTRACT_DECK and 'Deck' in combined_df.columns:
        deck_le_for_age = LabelEncoder()
        combined_df['Deck_Encoded_For_Age'] = deck_le_for_age.fit_transform(combined_df['Deck'])

    # 5. Fare の前処理
    print("\n--- Step 5: Fare Processing ---")
    
    # Fare の欠損値をクラス別中央値で補完
    fare_median_by_class = combined_df.groupby('Pclass')['Fare'].median()
    for pclass in [1, 2, 3]:
        mask = (combined_df['Pclass'] == pclass) & combined_df['Fare'].isnull()
        combined_df.loc[mask, 'Fare'] = fare_median_by_class[pclass]
    
    # 残りの欠損値があれば全体の中央値で補完
    combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())
    
    print("Fare missing values filled with class-specific medians")

    # 外れ値の処理（対数変換前）
    combined_df = handle_outliers(combined_df)
    
    # Fare の対数変換（設定に基づく）
    if config.FARE_LOG_TRANSFORM:
        combined_df['Fare'] = np.log1p(combined_df['Fare'])
        print("Applied log1p transformation to Fare")

    # 6. 高度なAge補完
    print("\n--- Step 6: Advanced Age Imputation ---")
    combined_df = advanced_age_imputation(combined_df)
    
    # 一時的なエンコード列を削除
    temp_columns_to_drop = ['Title_Encoded_For_Age']
    if config.CABIN_EXTRACT_DECK and 'Deck_Encoded_For_Age' in combined_df.columns:
        temp_columns_to_drop.append('Deck_Encoded_For_Age')
    
    existing_temp_drops = [col for col in temp_columns_to_drop if col in combined_df.columns]
    combined_df.drop(existing_temp_drops, axis=1, inplace=True)

    # 7. 統計的特徴量の作成
    print("\n--- Step 7: Statistical Feature Engineering ---")
    if config.CREATE_STATISTICAL_FEATURES:
        combined_df = create_statistical_features(combined_df)
    else:
        print("Statistical feature creation disabled in config.")

    # 8. 最終的なカテゴリ変数のエンコーディング
    print("\n--- Step 8: Final Categorical Encoding ---")
    
    # 設定に基づいて作成された特徴量も含めてエンコーディング
    categorical_columns = []
    
    # 基本的なカテゴリカル特徴量
    if config.CABIN_EXTRACT_DECK and 'Deck' in combined_df.columns:
        categorical_columns.append('Deck')
    if 'Title' in combined_df.columns:
        categorical_columns.append('Title')
    
    # 統計的特徴量から生成されたカテゴリカル特徴量
    if config.CREATE_STATISTICAL_FEATURES:
        for col in ['AgeGroup', 'FareGroup', 'FamilySizeGroup', 'TitleGroup']:
            if col in combined_df.columns:
                categorical_columns.append(col)
    
    for col in categorical_columns:
        if col in combined_df.columns:
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col].astype(str))
            print(f"Encoded {col}")

    # 9. 不要カラムの削除
    print("\n--- Step 9: Dropping Unnecessary Columns ---")
    columns_to_drop = ['Name', 'Ticket', 'Cabin']  # SibSp, Parchは残す
    
    # CabinNumberは設定で作成された場合のみ削除対象に追加
    if config.CABIN_EXTRACT_NUMBER and 'CabinNumber' in combined_df.columns:
        columns_to_drop.append('CabinNumber')
    
    existing_drops = [col for col in columns_to_drop if col in combined_df.columns]
    combined_df.drop(existing_drops, axis=1, inplace=True)
    print(f"Dropped columns: {existing_drops}")

    # 10. データ品質の検証
    print("\n--- Step 10: Data Quality Validation ---")
    if config.VALIDATE_DATA_QUALITY:
        validate_data_quality(combined_df)
    else:
        print("Data quality validation disabled in config.")

    # 11. データをtrainとtestに再分割
    print("\n--- Step 11: Splitting Data ---")
    train_processed_df = combined_df.loc['train'].set_index(original_train_index)
    test_processed_df = combined_df.loc['test'].set_index(original_test_index)
    
    # PassengerIdを除く特徴量を選択
    features_to_use = [col for col in train_processed_df.columns if col != 'PassengerId']

    X = train_processed_df[features_to_use]
    y = y_train_original
    X_test = test_processed_df[features_to_use]
    
    print(f"\nFinal features for training: {X.columns.tolist()}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}, Shape of X_test: {X_test.shape}")
    
    # 最終的なデータ型確認
    print("\nFinal data types:")
    print(X.dtypes)
    
    print("\n=== Enhanced Preprocessing Completed ===")
    
    return X, y, X_test 