#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量選択結果を正確に再現した最終提出ファイル作成スクリプト
AUC 0.8831を再現
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import sys
from datetime import datetime

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.utils import get_project_root, seed_everything
from src import config

def create_final_submission():
    """特徴量選択結果を正確に再現した最終提出ファイル作成"""
    
    print("=== Creating Final Optimized Submission ===")
    print("目標: AUC 0.8831の再現")
    
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
    
    print(f"使用する最適特徴量: {len(SELECTED_FEATURES)}個")
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
    
    # 特徴量選択時と同じ分割方法で性能確認
    print("\n特徴量選択時と同じ条件で性能確認...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_processed, 
        test_size=0.3,  # 特徴量選択時と同じ
        random_state=config.RANDOM_STATE, 
        stratify=y_processed
    )
    
    # 特徴量選択時と同じモデル設定
    rf_model = RandomForestClassifier(
        n_estimators=30,  # 特徴量選択時と同じ軽量設定
        random_state=config.RANDOM_STATE
    )
    
    rf_model.fit(X_tr, y_tr)
    val_pred = rf_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"検証AUC: {val_auc:.4f}")
    print(f"目標AUC 0.8831との差: {(val_auc - 0.8831)*100:+.2f}%")
    
    # より良いモデルで最終予測（提出用）
    print("\n最終予測用モデル訓練...")
    final_model = RandomForestClassifier(
        n_estimators=200,  # 提出用により良い設定
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    final_model.fit(X_train, y_processed)
    
    # 特徴量重要度
    print("\n特徴量重要度:")
    importance_df = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"  {i+1}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # テストデータでの予測
    print("\nテストデータでの最終予測...")
    test_predictions = final_model.predict_proba(X_test)[:, 1]
    
    # PassengerIdの取得
    passenger_ids = test_df['PassengerId'].values
    
    # 提出用DataFrame作成
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (test_predictions > 0.5).astype(int)
    })
    
    # ルートレベルのresultsディレクトリに保存
    root_results_dir = project_root / 'results'
    root_results_dir.mkdir(exist_ok=True)
    
    # タイムスタンプ付きファイル名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    submission_filename = root_results_dir / f'submission_optimized_{timestamp}.csv'
    
    # メインの提出ファイル
    submission_df.to_csv(submission_filename, index=False)
    
    # 現在時刻でのベスト提出ファイル（上書き用）
    best_submission_filename = root_results_dir / 'submission_best_features.csv'
    submission_df.to_csv(best_submission_filename, index=False)
    
    # 確率値付きファイル
    submission_with_proba = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (test_predictions > 0.5).astype(int),
        'Probability': test_predictions
    })
    proba_filename = root_results_dir / f'submission_with_proba_{timestamp}.csv'
    submission_with_proba.to_csv(proba_filename, index=False)
    
    # 詳細サマリー
    summary_filename = root_results_dir / f'submission_summary_{timestamp}.txt'
    survival_rate = submission_df['Survived'].mean()
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("=== FINAL OPTIMIZED SUBMISSION ===\n\n")
        f.write(f"特徴量選択による最適化結果\n")
        f.write(f"検証AUC: {val_auc:.4f}\n")
        f.write(f"特徴量数: {len(SELECTED_FEATURES)} (元20個から最適化)\n")
        f.write(f"予測生存率: {survival_rate:.1%}\n")
        f.write(f"生存予測数: {submission_df['Survived'].sum()}/{len(submission_df)}\n\n")
        
        f.write("=== パフォーマンス改善履歴 ===\n")
        f.write("1. ベースライン: 0.775 (公開リーダーボード)\n")
        f.write("2. 前処理改善: 0.8+ (大幅改善)\n")
        f.write("3. 特徴量選択: 0.8831 (さらに改善)\n\n")
        
        f.write("使用特徴量 (重要度順):\n")
        for i, (_, row) in enumerate(importance_df.iterrows()):
            f.write(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}\n")
        
        f.write(f"\n=== ファイル情報 ===\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"提出ファイル: {submission_filename.name}\n")
        f.write(f"ベストファイル: {best_submission_filename.name}\n")
    
    print(f"\n=== 最終提出ファイル作成完了 ===")
    print(f"📁 保存場所: {root_results_dir}")
    print(f"📄 提出用CSV: {submission_filename.name}")
    print(f"📄 ベストCSV: {best_submission_filename.name}")
    print(f"📄 確率付きCSV: {proba_filename.name}")
    print(f"📄 サマリー: {summary_filename.name}")
    
    print(f"\n📊 最終結果:")
    print(f"  検証AUC: {val_auc:.4f}")
    print(f"  予測生存率: {survival_rate:.1%}")
    print(f"  生存予測: {submission_df['Survived'].sum()}人 / {len(submission_df)}人")
    
    # 提出用CSVの一部を表示
    print(f"\n📋 提出用CSV (最初の10行):")
    print(submission_df.head(10))
    
    print(f"\n🎯 推奨提出ファイル: {best_submission_filename}")
    
    return submission_df, val_auc, str(best_submission_filename)

if __name__ == "__main__":
    submission_df, auc_score, best_file = create_final_submission()
    print(f"\n✅ 最終AUC: {auc_score:.4f}")
    print(f"✅ Kaggle提出推奨ファイル: {best_file}") 