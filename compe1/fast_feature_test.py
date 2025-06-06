#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速特徴量選択テスト
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def fast_feature_test():
    """高速特徴量選択テスト"""
    
    print("=== Fast Feature Selection Test ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # データの準備
    print("データの読み込みと前処理...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X, y, _ = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"前処理後: {X.shape[1]}特徴量")
    
    # 単純な訓練/検証分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"訓練: {X_train.shape[0]}サンプル, 検証: {X_val.shape[0]}サンプル")
    
    # ベースライン評価 (少ない木)
    print("\nベースライン評価...")
    rf_base = RandomForestClassifier(n_estimators=30, random_state=config.RANDOM_STATE)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict_proba(X_val)[:, 1]
    base_auc = roc_auc_score(y_val, y_pred_base)
    
    print(f"ベースライン AUC: {base_auc:.4f}")
    
    # 特徴量重要度ベース選択 (RandomForest)
    print("\n重要度ベース特徴量選択...")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_base.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("上位10特徴量:")
    for i in range(10):
        feat = importance_df.iloc[i]['feature']
        imp = importance_df.iloc[i]['importance']
        print(f"  {i+1:2d}. {feat:25s}: {imp:.4f}")
    
    # 上位K特徴量での評価
    results = []
    for k in [8, 10, 12, 15]:
        top_k_features = importance_df.head(k)['feature'].tolist()
        X_train_k = X_train[top_k_features]
        X_val_k = X_val[top_k_features]
        
        rf_k = RandomForestClassifier(n_estimators=30, random_state=config.RANDOM_STATE)
        rf_k.fit(X_train_k, y_train)
        y_pred_k = rf_k.predict_proba(X_val_k)[:, 1]
        auc_k = roc_auc_score(y_val, y_pred_k)
        
        improvement = (auc_k - base_auc) / base_auc * 100
        results.append({
            'k': k,
            'auc': auc_k,
            'improvement': improvement,
            'features': top_k_features
        })
        
        print(f"上位{k:2d}特徴量: AUC = {auc_k:.4f}, 改善率 = {improvement:+.2f}%")
    
    # F統計量ベース選択との比較
    print("\nF統計量ベース選択...")
    selector = SelectKBest(f_classif, k=10)
    X_train_f = selector.fit_transform(X_train, y_train)
    X_val_f = selector.transform(X_val)
    selected_features_f = X.columns[selector.get_support()].tolist()
    
    rf_f = RandomForestClassifier(n_estimators=30, random_state=config.RANDOM_STATE)
    rf_f.fit(X_train_f, y_train)
    y_pred_f = rf_f.predict_proba(X_val_f)[:, 1]
    auc_f = roc_auc_score(y_val, y_pred_f)
    improvement_f = (auc_f - base_auc) / base_auc * 100
    
    print(f"F統計量選択: AUC = {auc_f:.4f}, 改善率 = {improvement_f:+.2f}%")
    print(f"選択特徴量: {selected_features_f}")
    
    # 最良結果の特定
    best_result = max(results, key=lambda x: x['auc'])
    
    print("\n" + "="*60)
    print("結果サマリー")
    print("="*60)
    print(f"ベースライン ({X.shape[1]}特徴量): AUC = {base_auc:.4f}")
    print(f"重要度選択 最良 ({best_result['k']}特徴量): AUC = {best_result['auc']:.4f} ({best_result['improvement']:+.2f}%)")
    print(f"F統計量選択 (10特徴量): AUC = {auc_f:.4f} ({improvement_f:+.2f}%)")
    
    print(f"\n推奨特徴量セット ({best_result['k']}特徴量):")
    for i, feat in enumerate(best_result['features']):
        print(f"  {i+1:2d}. {feat}")
    
    # 結果保存
    os.makedirs('results', exist_ok=True)
    with open('results/fast_feature_selection.txt', 'w', encoding='utf-8') as f:
        f.write("=== FAST FEATURE SELECTION RESULTS ===\n\n")
        f.write(f"ベースライン: AUC = {base_auc:.4f}\n")
        f.write(f"最良手法: 重要度ベース {best_result['k']}特徴量\n")
        f.write(f"最良AUC: {best_result['auc']:.4f}\n")
        f.write(f"改善率: {best_result['improvement']:+.2f}%\n\n")
        f.write("推奨特徴量:\n")
        for i, feat in enumerate(best_result['features']):
            f.write(f"{i+1:2d}. {feat}\n")
    
    print(f"\n結果保存: results/fast_feature_selection.txt")
    
    return best_result['features'], best_result['auc']

if __name__ == "__main__":
    features, score = fast_feature_test()
    print(f"\n最終結果: AUC = {score:.4f}")
    print(f"特徴量数: {len(features)}") 