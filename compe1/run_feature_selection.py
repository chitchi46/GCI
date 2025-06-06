#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量選択の実行スクリプト
改善された前処理済みデータに対して最適な特徴量セットを見つける
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import argparse

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_selection import FeatureSelector
from src.trainer import train_model
from src.utils import get_project_root, seed_everything, save_submission_file
from src import config

def main():
    """特徴量選択のメイン処理"""
    parser = argparse.ArgumentParser(description="Run feature selection for improved preprocessing pipeline.")
    parser.add_argument("--quick", action="store_true", help="Run quick feature selection with basic methods only.")
    args = parser.parse_args()
    
    start_time = time.time()
    seed_everything(config.RANDOM_STATE)
    
    print("=" * 80)
    print("FEATURE SELECTION FOR TITANIC SURVIVAL PREDICTION")
    print("=" * 80)
    
    # プロジェクトルートの取得
    project_root = get_project_root()
    
    # 1. データの読み込み
    print("\n--- 1. データの読み込み ---")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    # 2. 改善された前処理の実行
    print("\n--- 2. 改善された前処理の実行 ---")
    X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"Processed features: {len(X_processed.columns)}")
    print(f"Features: {list(X_processed.columns)}")
    
    # 3. 特徴量選択の実行
    print("\n--- 3. 特徴量選択の実行 ---")
    selector = FeatureSelector(random_state=config.RANDOM_STATE)
    
    # 基本的な特徴量選択を実行
    results_df = selector.run_basic_selection(X_processed, y_processed)
    
    # 4. 最適な特徴量セットの特定
    print("\n--- 4. 最適な特徴量セットの特定 ---")
    
    # 最高性能の手法を選択
    best_method = results_df.iloc[0]
    best_features = best_method['features']
    
    print(f"\nBest method: {best_method['method']}")
    print(f"Number of features: {best_method['n_features']}")
    print(f"LightGBM CV AUC: {best_method['lgb_cv_mean']:.4f} (+/- {best_method['lgb_cv_std']*2:.4f})")
    print(f"RandomForest CV AUC: {best_method['rf_cv_mean']:.4f} (+/- {best_method['rf_cv_std']*2:.4f})")
    print(f"Selected features: {best_features}")
    
    # 5. 最適特徴量でのモデル学習
    print("\n--- 5. 最適特徴量でのモデル学習 ---")
    
    X_train_selected = X_processed[best_features]
    X_test_selected = X_test_processed[best_features]
    
    print(f"Training with {len(best_features)} selected features...")
    
    cv_models, oof_preds, test_preds, cv_score, cv_auc_score, cv_logloss_score, feature_importance_df = train_model(
        X_train_selected, 
        y_processed, 
        X_test_selected, 
        params=config.LGB_PARAMS,
        n_splits=config.N_SPLITS_CV, 
        random_seed=config.RANDOM_STATE,
        n_repeats=3
    )
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"CV Accuracy: {cv_score:.4f}")
    print(f"CV AUC: {cv_auc_score:.4f}")
    print(f"CV LogLoss: {cv_logloss_score:.4f}")
    
    # 6. 特徴量重要度の表示
    print("\n--- 6. 特徴量重要度 (選択された特徴量) ---")
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print("Top selected features by importance:")
    for i, (feature, importance) in enumerate(mean_importance.items()):
        print(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
    
    # 7. 提出ファイルの作成
    print("\n--- 7. 提出ファイルの作成 ---")
    current_time_str = time.strftime("%Y%m%d%H%M%S")
    experiment_id = f"feature_selected_{current_time_str}"
    
    submission_path = save_submission_file(test_df, test_preds, experiment_id)
    print(f"Submission file created: {submission_path}")
    
    # 8. 結果の要約保存
    print("\n--- 8. 結果の要約保存 ---")
    
    # 特徴量選択結果を保存
    results_path = project_root / "results" / f"feature_selection_results_{current_time_str}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Feature selection results saved: {results_path}")
    
    # 選択された特徴量リストを保存
    selected_features_path = project_root / "results" / f"selected_features_{current_time_str}.txt"
    with open(selected_features_path, 'w') as f:
        f.write(f"Best method: {best_method['method']}\n")
        f.write(f"CV AUC: {best_method['lgb_cv_mean']:.4f}\n")
        f.write(f"Number of features: {len(best_features)}\n\n")
        f.write("Selected features:\n")
        for feature in best_features:
            f.write(f"- {feature}\n")
    
    print(f"Selected features list saved: {selected_features_path}")
    
    # 9. 改善の比較
    print("\n--- 9. 改善の比較 ---")
    baseline_auc = results_df[results_df['method'] == 'baseline_all']['lgb_cv_mean'].iloc[0]
    best_auc = best_method['lgb_cv_mean']
    improvement = best_auc - baseline_auc
    
    print(f"Baseline (all features): {baseline_auc:.4f}")
    print(f"Best selection: {best_auc:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement/baseline_auc*100:+.2f}%)")
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main() 