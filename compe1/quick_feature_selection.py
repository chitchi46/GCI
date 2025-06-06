#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速特徴量選択スクリプト（軽量版）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def quick_feature_selection():
    """高速特徴量選択"""
    
    print("=== Quick Feature Selection ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # データの準備
    print("データの読み込み...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X, y, _ = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"前処理後の特徴量数: {X.shape[1]}")
    print(f"特徴量: {list(X.columns)}\n")
    
    # ベースラインの評価
    rf_base = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    base_scores = cross_val_score(rf_base, X, y, cv=3, scoring='roc_auc')
    base_mean = base_scores.mean()
    
    print(f"ベースライン (全特徴量): AUC = {base_mean:.4f} (+/- {base_scores.std()*2:.4f})")
    
    results = []
    results.append({
        'method': 'baseline_all',
        'n_features': len(X.columns),
        'features': list(X.columns),
        'cv_mean': base_mean,
        'cv_std': base_scores.std()
    })
    
    # 1. 単変量特徴選択 (上位10個)
    print("\n1. 単変量特徴選択 (F統計量, 上位10個)...")
    selector_f = SelectKBest(f_classif, k=10)
    X_f = selector_f.fit_transform(X, y)
    selected_features_f = X.columns[selector_f.get_support()].tolist()
    
    rf_f = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    scores_f = cross_val_score(rf_f, X_f, y, cv=3, scoring='roc_auc')
    
    print(f"F統計量 (10特徴量): AUC = {scores_f.mean():.4f} (+/- {scores_f.std()*2:.4f})")
    print(f"選択された特徴量: {selected_features_f}")
    
    results.append({
        'method': 'f_classif_k10',
        'n_features': len(selected_features_f),
        'features': selected_features_f,
        'cv_mean': scores_f.mean(),
        'cv_std': scores_f.std()
    })
    
    # 2. 相互情報量 (上位10個)
    print("\n2. 相互情報量選択 (上位10個)...")
    selector_mi = SelectKBest(mutual_info_classif, k=10)
    X_mi = selector_mi.fit_transform(X, y)
    selected_features_mi = X.columns[selector_mi.get_support()].tolist()
    
    rf_mi = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    scores_mi = cross_val_score(rf_mi, X_mi, y, cv=3, scoring='roc_auc')
    
    print(f"相互情報量 (10特徴量): AUC = {scores_mi.mean():.4f} (+/- {scores_mi.std()*2:.4f})")
    print(f"選択された特徴量: {selected_features_mi}")
    
    results.append({
        'method': 'mutual_info_k10',
        'n_features': len(selected_features_mi),
        'features': selected_features_mi,
        'cv_mean': scores_mi.mean(),
        'cv_std': scores_mi.std()
    })
    
    # 3. RFE (上位12個)
    print("\n3. 再帰的特徴除去 (RFE, 12特徴量)...")
    estimator = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
    selector_rfe = RFE(estimator, n_features_to_select=12)
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[selector_rfe.get_support()].tolist()
    
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    scores_rfe = cross_val_score(rf_rfe, X_rfe, y, cv=3, scoring='roc_auc')
    
    print(f"RFE (12特徴量): AUC = {scores_rfe.mean():.4f} (+/- {scores_rfe.std()*2:.4f})")
    print(f"選択された特徴量: {selected_features_rfe}")
    
    results.append({
        'method': 'rfe_k12',
        'n_features': len(selected_features_rfe),
        'features': selected_features_rfe,
        'cv_mean': scores_rfe.mean(),
        'cv_std': scores_rfe.std()
    })
    
    # 4. 重要度ベース選択 (上位12個)
    print("\n4. RandomForest重要度ベース選択 (上位12個)...")
    rf_importance = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    rf_importance.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_importance.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(12)['feature'].tolist()
    X_importance = X[top_features]
    
    rf_imp = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    scores_imp = cross_val_score(rf_imp, X_importance, y, cv=3, scoring='roc_auc')
    
    print(f"重要度ベース (12特徴量): AUC = {scores_imp.mean():.4f} (+/- {scores_imp.std()*2:.4f})")
    print(f"選択された特徴量: {top_features}")
    print("\n特徴量重要度:")
    for i in range(min(8, len(top_features))):
        feat_name = importance_df.iloc[i]['feature']
        feat_imp = importance_df.iloc[i]['importance']
        print(f"  {i+1:2d}. {feat_name:25s}: {feat_imp:.4f}")
    
    results.append({
        'method': 'rf_importance_k12',
        'n_features': len(top_features),
        'features': top_features,
        'cv_mean': scores_imp.mean(),
        'cv_std': scores_imp.std()
    })
    
    # 結果の整理と表示
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_mean', ascending=False)
    
    print("\n" + "="*80)
    print("特徴量選択結果サマリー")
    print("="*80)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        improvement = (row['cv_mean'] - base_mean) / base_mean * 100
        print(f"{i+1}. {row['method']:20s}: {row['n_features']:2d}特徴量, "
              f"AUC = {row['cv_mean']:.4f}, 改善率 = {improvement:+.2f}%")
    
    # 最良の手法
    best_method = results_df.iloc[0]
    print(f"\n最良の手法: {best_method['method']}")
    print(f"特徴量数: {best_method['n_features']}")
    print(f"AUC: {best_method['cv_mean']:.4f} (+/- {best_method['cv_std']*2:.4f})")
    print(f"改善率: {((best_method['cv_mean'] - base_mean) / base_mean * 100):+.2f}%")
    print(f"選択された特徴量: {best_method['features']}")
    
    # 結果の保存
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/quick_feature_selection_results.csv', index=False)
    
    with open('results/quick_feature_selection_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=== QUICK FEATURE SELECTION RESULTS ===\n\n")
        f.write(f"ベースライン (全{X.shape[1]}特徴量): AUC = {base_mean:.4f}\n\n")
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            improvement = (row['cv_mean'] - base_mean) / base_mean * 100
            f.write(f"{i+1}. {row['method']}\n")
            f.write(f"   特徴量数: {row['n_features']}\n")
            f.write(f"   AUC: {row['cv_mean']:.4f} (+/- {row['cv_std']*2:.4f})\n")
            f.write(f"   改善率: {improvement:+.2f}%\n")
            f.write(f"   特徴量: {row['features']}\n\n")
        
        f.write(f"BEST METHOD: {best_method['method']}\n")
        f.write(f"BEST FEATURES: {best_method['features']}\n")
    
    print(f"\n結果保存先:")
    print(f"- results/quick_feature_selection_results.csv")
    print(f"- results/quick_feature_selection_summary.txt")
    
    return results_df, best_method['features']

if __name__ == "__main__":
    results_df, best_features = quick_feature_selection()