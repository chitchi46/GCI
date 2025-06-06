#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルな特徴量選択テスト
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def simple_test():
    """シンプルなテスト"""
    
    print("=== Simple Feature Selection Test ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # データの準備
    print("データの読み込みと前処理...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X, y, _ = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"前処理後: {X.shape[1]}特徴量")
    print(f"特徴量: {list(X.columns)}")
    
    # ベースライン評価
    print("\nベースライン評価 (2-fold CV, 50 trees)...")
    rf_base = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
    base_scores = cross_val_score(rf_base, X, y, cv=2, scoring='roc_auc')
    base_mean = base_scores.mean()
    
    print(f"ベースライン AUC: {base_mean:.4f}")
    
    # 上位10特徴量の選択
    print("\n上位10特徴量の選択...")
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"選択された特徴量: {selected_features}")
    
    # 選択後の評価
    print("\n選択後の評価...")
    rf_selected = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
    selected_scores = cross_val_score(rf_selected, X_selected, y, cv=2, scoring='roc_auc')
    selected_mean = selected_scores.mean()
    
    print(f"選択後 AUC: {selected_mean:.4f}")
    
    improvement = (selected_mean - base_mean) / base_mean * 100
    print(f"改善率: {improvement:+.2f}%")
    
    return selected_features, selected_mean

if __name__ == "__main__":
    features, score = simple_test()
    print(f"\n最終結果:")
    print(f"選択特徴量: {features}")
    print(f"AUCスコア: {score:.4f}") 