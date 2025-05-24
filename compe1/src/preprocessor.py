# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from src import config # config.py から定数をインポート

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    データの前処理を行う関数 (EDAに基づき実装)
    Ageは線形回帰で補完、CabinからはDeck情報を抽出
    """
    print("Starting preprocessing...")
    
    # ターゲットカラムを学習データから一時的に分離し、後で結合するために保持
    y_train_original = train_df[config.TARGET_COLUMN]
    train_df_no_target = train_df.drop(config.TARGET_COLUMN, axis=1)

    combined_df = pd.concat([train_df_no_target, test_df], keys=['train', 'test'])
    original_train_index = train_df.index
    original_test_index = test_df.index

    # 1. Name から Title を抽出 & マッピング (既存のものを活用)
    combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss", "Ms": "Miss",
        "Sir": "Rare", "Lady": "Rare", "Mme": "Mrs", "Capt": "Rare", "Countess": "Rare",
        "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "the Countess": "Rare"
    }
    combined_df['Title'] = combined_df['Title'].map(title_mapping).fillna('Unknown')

    # 2. Cabin から Deck を抽出
    combined_df['Deck'] = combined_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M') # M for Missing

    # 3. カテゴリ変数のエンコーディング (Age補完の前に必要なもの)
    # Sex のエンコード
    combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Embarked の欠損値を最頻値で補完 & エンコード
    combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    combined_df['Embarked'] = combined_df['Embarked'].map(embarked_mapping)
    
    # Title のエンコード (Age補完のためにLabelEncoderを使用)
    title_le_for_age = LabelEncoder()
    combined_df['Title_Encoded_For_Age'] = title_le_for_age.fit_transform(combined_df['Title'])

    # Deck のエンコード (Age補完のためにLabelEncoderを使用)
    deck_le_for_age = LabelEncoder()
    combined_df['Deck_Encoded_For_Age'] = deck_le_for_age.fit_transform(combined_df['Deck'])

    # 4. Fareの欠損値を中央値で補完 (テストデータに1件ある)
    combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)

    # 5. Age の欠損値補完 (モデルベース - 線形回帰)
    print("Imputing Age using Linear Regression...")
    age_df = combined_df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title_Encoded_For_Age', 'Deck_Encoded_For_Age']].copy()
    
    age_known = age_df[age_df['Age'].notnull()]
    age_unknown = age_df[age_df['Age'].isnull()]

    if not age_unknown.empty and not age_known.empty(): # age_knownも空でないことを確認
        X_age_train = age_known.drop('Age', axis=1)
        y_age_train = age_known['Age']
        X_age_predict = age_unknown.drop('Age', axis=1)

        # Age予測に使用する特徴量に欠損がないか確認 (Fareなどで発生しうる)
        # もしX_age_trainやX_age_predictにNaNが残っている場合、補完するか、その行/列を除外する必要がある
        # ここでは、上記までの処理でこれらの特徴量のNaNは処理済みと仮定
        if X_age_train.isnull().any().any() or X_age_predict.isnull().any().any():
            print("Warning: NaN found in features for Age imputation. Filling with median for safety.")
            for col in X_age_train.columns:
                if X_age_train[col].isnull().any():
                    median_val = X_age_train[col].median() # Fit on train part only
                    X_age_train[col].fillna(median_val, inplace=True)
                    if col in X_age_predict.columns:
                         X_age_predict[col].fillna(median_val, inplace=True)

        lr_age = LinearRegression()
        lr_age.fit(X_age_train, y_age_train)
        
        predicted_age = lr_age.predict(X_age_predict)
        
        predicted_age = np.clip(predicted_age, 0, 80)
        
        combined_df.loc[combined_df['Age'].isnull(), 'Age'] = predicted_age
        print("Age imputation finished.")
    elif age_known.empty:
        print("Warning: No data with known Age to train imputation model. Filling Age with overall median.")
        combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True) 
    else:
        print("No missing Age values to impute or no data to predict for.")

    combined_df.drop(['Title_Encoded_For_Age', 'Deck_Encoded_For_Age'], axis=1, inplace=True)

    # 6. FamilySize と IsAlone を作成 (既存処理)
    combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
    combined_df['IsAlone'] = 0
    combined_df.loc[combined_df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # 7. 最終的なカテゴリ変数のエンコーディング
    final_title_le = LabelEncoder()
    combined_df['Title'] = final_title_le.fit_transform(combined_df['Title'])

    final_deck_le = LabelEncoder()
    combined_df['Deck'] = final_deck_le.fit_transform(combined_df['Deck'])

    # 8. 不要カラムの削除
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    combined_df.drop(columns_to_drop, axis=1, inplace=True)
    
    # 9. データをtrainとtestに再分割
    train_processed_df = combined_df.loc['train'].set_index(original_train_index)
    test_processed_df = combined_df.loc['test'].set_index(original_test_index)
    
    features_to_use = [col for col in train_processed_df.columns if col != 'PassengerId'] 

    X = train_processed_df[features_to_use]
    y = y_train_original
    X_test = test_processed_df[features_to_use]
    
    print(f"Features for training: {X.columns.tolist()}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}, Shape of X_test: {X_test.shape}")
    print("Preprocessing finished.")
    return X, y, X_test 