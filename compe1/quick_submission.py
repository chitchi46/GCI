#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速提出用CSV作成スクリプト
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def quick_submission():
    """高速提出用CSV作成"""
    
    print("=== Quick Submission Creation ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # 最適特徴量セット（特徴量選択結果から）
    SELECTED_FEATURES = [
        'Sex_Age_Interaction',
        'Pclass_Fare_Interaction', 
        'Age_Fare_Interaction',
        'Age',
        'Fare',
        'Title',
        'Sex',
        'Pclass'
    ]
    
    print(f"使用する特徴量: {len(SELECTED_FEATURES)}個")
    for i, feat in enumerate(SELECTED_FEATURES, 1):
        print(f"  {i}. {feat}")
    
    # データの読み込みと前処理
    print("\nデータの読み込みと前処理...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
    
    # 選択された特徴量のみを使用
    X_train = X_processed[SELECTED_FEATURES]
    X_test = X_test_processed[SELECTED_FEATURES]
    
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    
    # 簡単な検証分割での性能確認
    print("\n簡易性能確認...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_processed, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_processed
    )
    
    # モデル訓練（軽量設定）
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    rf_model.fit(X_tr, y_tr)
    val_pred = rf_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"検証AUC: {val_auc:.4f}")
    
    # 全データでモデル再訓練
    print("\n全データでモデル訓練...")
    rf_model.fit(X_train, y_processed)
    
    # 特徴量重要度
    print("\n特徴量重要度:")
    importance_df = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"  {i+1}. {row['feature']:20s}: {row['importance']:.3f}")
    
    # テストデータでの予測
    print("\nテストデータでの予測...")
    test_predictions = rf_model.predict_proba(X_test)[:, 1]
    
    # PassengerIdの取得
    passenger_ids = test_df['PassengerId'].values
    
    # 提出用DataFrame作成
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (test_predictions > 0.5).astype(int)
    })
    
    # 結果保存
    os.makedirs('results', exist_ok=True)
    
    # CSVファイル保存
    submission_filename = 'results/submission_optimized.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    # 確率値付きファイル
    submission_with_proba = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (test_predictions > 0.5).astype(int),
        'Probability': test_predictions
    })
    proba_filename = 'results/submission_with_proba.csv'
    submission_with_proba.to_csv(proba_filename, index=False)
    
    # 統計情報
    survival_rate = submission_df['Survived'].mean()
    print(f"\n=== 提出ファイル作成完了 ===")
    print(f"提出用CSV: {submission_filename}")
    print(f"確率付きCSV: {proba_filename}")
    print(f"検証AUC: {val_auc:.4f}")
    print(f"予測生存率: {survival_rate:.1%}")
    print(f"生存予測: {submission_df['Survived'].sum()}人 / {len(submission_df)}人")
    
    # 最初の10行を表示
    print(f"\n提出用CSV (最初の10行):")
    print(submission_df.head(10))
    
    # サマリーファイル
    with open('results/quick_submission_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=== QUICK SUBMISSION SUMMARY ===\n\n")
        f.write(f"検証AUC: {val_auc:.4f}\n")
        f.write(f"特徴量数: {len(SELECTED_FEATURES)}\n")
        f.write(f"予測生存率: {survival_rate:.1%}\n")
        f.write(f"生存予測数: {submission_df['Survived'].sum()}/{len(submission_df)}\n\n")
        f.write("使用特徴量:\n")
        for i, feat in enumerate(SELECTED_FEATURES, 1):
            f.write(f"{i}. {feat}\n")
        f.write("\n特徴量重要度:\n")
        for i, (_, row) in enumerate(importance_df.iterrows()):
            f.write(f"{i+1}. {row['feature']:20s}: {row['importance']:.3f}\n")
    
    print(f"サマリー: results/quick_submission_summary.txt")
    
    return submission_df, val_auc

if __name__ == "__main__":
    submission_df, auc_score = quick_submission()
    print(f"\n最終結果: AUC = {auc_score:.4f}")
    print(f"提出ファイル: results/submission_optimized.csv") 