#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量選択結果を使用した提出用CSV作成スクリプト
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def create_submission_with_selected_features():
    """特徴量選択結果を使用した提出用CSV作成"""
    
    print("=== Creating Submission with Optimized Features ===")
    
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
    
    print(f"使用する特徴量 ({len(SELECTED_FEATURES)}個):")
    for i, feat in enumerate(SELECTED_FEATURES, 1):
        print(f"  {i:2d}. {feat}")
    
    # データの読み込みと前処理
    print("\nデータの読み込みと前処理...")
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
    
    print(f"前処理後の特徴量数: {X_processed.shape[1]}")
    print(f"利用可能な特徴量: {list(X_processed.columns)}")
    
    # 選択された特徴量のみを使用
    missing_features = [feat for feat in SELECTED_FEATURES if feat not in X_processed.columns]
    if missing_features:
        print(f"警告: 以下の特徴量が見つかりません: {missing_features}")
        available_features = [feat for feat in SELECTED_FEATURES if feat in X_processed.columns]
        print(f"利用可能な特徴量のみ使用: {available_features}")
        SELECTED_FEATURES = available_features
    
    X_train = X_processed[SELECTED_FEATURES]
    X_test = X_test_processed[SELECTED_FEATURES]
    
    print(f"\n使用する特徴量: {X_train.shape[1]}個")
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    
    # クロスバリデーションでの性能確認
    print("\nクロスバリデーションでの性能確認...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf_model, X_train, y_processed, cv=5, scoring='roc_auc')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"5-fold CV AUC: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    print(f"CV scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # 全データでモデル訓練
    print("\n全データでモデル訓練...")
    rf_model.fit(X_train, y_processed)
    
    # 特徴量重要度の表示
    print("\n特徴量重要度:")
    importance_df = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"  {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # テストデータでの予測
    print("\nテストデータでの予測...")
    test_predictions = rf_model.predict_proba(X_test)[:, 1]
    
    print(f"予測統計:")
    print(f"  平均: {test_predictions.mean():.4f}")
    print(f"  標準偏差: {test_predictions.std():.4f}")
    print(f"  最小値: {test_predictions.min():.4f}")
    print(f"  最大値: {test_predictions.max():.4f}")
    
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
    submission_filename = f'results/submission_optimized_features.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    # 確率値付きファイルも保存
    submission_with_proba = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (test_predictions > 0.5).astype(int),
        'Probability': test_predictions
    })
    proba_filename = f'results/submission_with_probability.csv'
    submission_with_proba.to_csv(proba_filename, index=False)
    
    # サマリーファイル作成
    summary_filename = 'results/submission_summary.txt'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("=== SUBMISSION SUMMARY ===\n\n")
        f.write(f"使用モデル: RandomForest\n")
        f.write(f"特徴量数: {len(SELECTED_FEATURES)}\n")
        f.write(f"CV AUC: {cv_mean:.4f} (+/- {cv_std*2:.4f})\n\n")
        f.write("使用特徴量:\n")
        for i, feat in enumerate(SELECTED_FEATURES, 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("\n特徴量重要度:\n")
        for i, (_, row) in enumerate(importance_df.iterrows()):
            f.write(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}\n")
        f.write(f"\n予測統計:\n")
        f.write(f"平均: {test_predictions.mean():.4f}\n")
        f.write(f"標準偏差: {test_predictions.std():.4f}\n")
        f.write(f"生存予測数: {submission_df['Survived'].sum()}/{len(submission_df)} ({submission_df['Survived'].sum()/len(submission_df)*100:.1f}%)\n")
    
    print(f"\n=== 提出ファイル作成完了 ===")
    print(f"提出用CSV: {submission_filename}")
    print(f"確率付きCSV: {proba_filename}")
    print(f"サマリー: {summary_filename}")
    print(f"\n予測結果:")
    print(f"  生存予測: {submission_df['Survived'].sum()}人")
    print(f"  死亡予測: {len(submission_df) - submission_df['Survived'].sum()}人")
    print(f"  生存率: {submission_df['Survived'].sum()/len(submission_df)*100:.1f}%")
    
    # 提出用CSVの一部を表示
    print(f"\n提出用CSV (最初の10行):")
    print(submission_df.head(10))
    
    return submission_df, cv_mean

if __name__ == "__main__":
    submission_df, cv_score = create_submission_with_selected_features()
    print(f"\n最終CV AUC: {cv_score:.4f}")
    print(f"提出ファイル: results/submission_optimized_features.csv") 