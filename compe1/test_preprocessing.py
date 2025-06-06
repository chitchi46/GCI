#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前処理の改善をテストするためのスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root
from src import config

def test_preprocessing():
    """前処理のテストを実行"""
    print("=== Testing Enhanced Preprocessing ===")
    
    # プロジェクトルートの取得
    project_root = get_project_root()
    
    # データの読み込み
    print("\n1. Loading data...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    # 前処理前のデータ概要
    print("\n2. Before preprocessing:")
    print("Train missing values:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
    print("\nTest missing values:")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0])
    
    # 前処理の実行
    print("\n3. Running enhanced preprocessing...")
    try:
        X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
        
        print(f"\nProcessed train features shape: {X_processed.shape}")
        print(f"Processed train target shape: {y_processed.shape}")
        print(f"Processed test features shape: {X_test_processed.shape}")
        
        # 前処理後のデータ概要
        print("\n4. After preprocessing:")
        print("Features created:")
        print(X_processed.columns.tolist())
        
        print("\nMissing values in processed data:")
        missing_train = X_processed.isnull().sum()
        missing_test = X_test_processed.isnull().sum()
        
        if missing_train.sum() > 0:
            print("Train missing values:")
            print(missing_train[missing_train > 0])
        else:
            print("No missing values in processed train data ✓")
            
        if missing_test.sum() > 0:
            print("Test missing values:")
            print(missing_test[missing_test > 0])
        else:
            print("No missing values in processed test data ✓")
        
        # データ型の確認
        print("\n5. Data types:")
        print(X_processed.dtypes)
        
        # 基本統計
        print("\n6. Basic statistics:")
        print(X_processed.describe())
        
        # 特徴量の数
        print(f"\n7. Total features: {len(X_processed.columns)}")
        
        # カテゴリカル特徴量の一意値数
        print("\n8. Categorical features unique counts:")
        categorical_features = X_processed.select_dtypes(include=['int64']).columns
        for col in categorical_features:
            unique_count = X_processed[col].nunique()
            if unique_count < 20:  # カテゴリカルと思われる特徴量
                print(f"{col}: {unique_count} unique values")
        
        print("\n=== Preprocessing test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_old_preprocessing():
    """旧前処理との比較（オプション）"""
    print("\n=== Comparing with Previous Preprocessing ===")
    # 必要に応じて旧前処理との比較を実装
    pass

if __name__ == "__main__":
    # テストの実行
    success = test_preprocessing()
    
    if success:
        print("\n✅ All preprocessing tests passed!")
    else:
        print("\n❌ Preprocessing tests failed!")
        sys.exit(1) 