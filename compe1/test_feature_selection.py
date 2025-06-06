#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量選択のテストスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_selection import FeatureSelector
from src.utils import get_project_root, seed_everything
from src import config

def test_feature_selection():
    """特徴量選択のテスト"""
    
    print("=== Feature Selection Test ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # データの読み込み
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 前処理
    print("\nRunning preprocessing...")
    X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"Processed train shape: {X_processed.shape}")
    print(f"Features: {list(X_processed.columns)}")
    
    # 特徴量選択
    print("\nRunning feature selection...")
    selector = FeatureSelector(random_state=config.RANDOM_STATE)
    
    # 基本的な特徴量選択のみ実行
    results_df = selector.run_basic_selection(X_processed, y_processed)
    
    print("\n=== Results Summary ===")
    print(results_df[['method', 'n_features', 'rf_cv_mean', 'lgb_cv_mean']].round(4))
    
    # 最良の手法
    best_method = results_df.iloc[0]
    print(f"\nBest method: {best_method['method']}")
    print(f"Number of features: {best_method['n_features']}")
    print(f"Features: {best_method['features']}")
    print(f"LightGBM CV AUC: {best_method['lgb_cv_mean']:.4f}")
    
    return results_df

if __name__ == "__main__":
    test_feature_selection() 