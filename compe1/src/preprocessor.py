# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src import config # config.py から定数をインポート

def preprocess_data(train_df, test_df):
    """
    データの前処理を行う関数 (EDAに基づき実装)
    """
    print("Starting preprocessing...")
    # train_dfとtest_dfを結合して一度に処理できるようにする（インデックスを保持）
    # ターゲットカラムは学習データにし存在しないため、先に削除してから結合する
    combined_df = pd.concat([train_df.drop(config.TARGET_COLUMN, axis=1), test_df], keys=['train', 'test'])
    original_train_index = train_df.index
    original_test_index = test_df.index

    # 1. Name から Title を抽出
    combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss", "Ms": "Miss",
        "Sir": "Rare", "Lady": "Rare", "Mme": "Mrs", "Capt": "Rare", "Countess": "Rare",
        "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "the Countess":"Rare" # the Countess を追加
    }
    combined_df['Title'] = combined_df['Title'].map(title_mapping).fillna('Unknown')

    # 2. FamilySize と IsAlone を作成
    combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
    combined_df['IsAlone'] = 0
    combined_df.loc[combined_df['FamilySize'] == 1, 'IsAlone'] = 1

    # 3. Age の欠損値補完 (Titleごとの中央値を使用)
    title_age_median = combined_df.groupby('Title')['Age'].median()
    for title_val in title_age_median.index:
        combined_df.loc[(combined_df['Age'].isnull()) & (combined_df['Title'] == title_val), 'Age'] = title_age_median[title_val]
    if combined_df['Age'].isnull().any(): # まだNaNが残っている場合 (例えばRare TitleでAgeが全てNaNだった場合など)
        combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True)

    # カテゴリ変数のエンコーディング
    # Sex のエンコード
    combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Embarked の欠損値を最頻値で補完
    combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)
    # Embarked のエンコード
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    combined_df['Embarked'] = combined_df['Embarked'].map(embarked_mapping)
    
    # Title のエンコード
    title_le = LabelEncoder()
    combined_df['Title'] = title_le.fit_transform(combined_df['Title'])

    # Fareの欠損値を中央値で補完 (テストデータに1件ある)
    combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)

    # 不要カラムの削除 (SibSp と Parch は FamilySize に集約されたため削除)
    combined_df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)
    
    # PassengerIdはインデックスとして扱うため、一度リセットしてから処理し、最後に再設定する
    # ただし、ここではcombined_dfのインデックスは(train/test, original_index)のMultiIndexになっている
    # PassengerIdは特徴量としては使わないので、この段階で処理する必要はない
    # test_dfの提出時にPassengerIdが必要になる

    # データをtrainとtestに再分割
    train_processed_df = combined_df.loc['train'].set_index(original_train_index)
    test_processed_df = combined_df.loc['test'].set_index(original_test_index)
    
    # ID_COLUMN (PassengerId) が特徴量に含まれていないことを確認
    # train_processed_df には元々 PassengerId が含まれていない想定 (targetと共にdropしたため)
    # test_processed_df には PassengerId が残っているはずだが、特徴量セットからは除外する
    features_to_use = [col for col in train_processed_df.columns if col != config.ID_COLUMN]
    # test_dfからも同じ特徴量セットを抽出 (PassengerIdは提出時に使うので残しておく)
    
    X = train_processed_df[features_to_use]
    y = train_df[config.TARGET_COLUMN] # 元のtrain_dfから取得 (preprocess_dataに渡された時点のもの)
    X_test = test_processed_df[features_to_use]
    
    print(f"Features for training: {X.columns.tolist()}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}, Shape of X_test: {X_test.shape}")
    print("Preprocessing finished.")
    return X, y, X_test 