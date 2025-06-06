#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量選択結果の可視化スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_selection import FeatureSelector
from src.utils import get_project_root, seed_everything
from src import config

def create_feature_selection_visualization():
    """特徴量選択結果の可視化"""
    
    print("=== Feature Selection Visualization ===")
    
    seed_everything(config.RANDOM_STATE)
    project_root = get_project_root()
    
    # データの準備
    train_df = load_train_data(project_root / config.TRAIN_DATA_PATH)
    test_df = load_test_data(project_root / config.TEST_DATA_PATH)
    X_processed, y_processed, _ = preprocess_data(train_df.copy(), test_df.copy())
    
    # 特徴量選択の実行
    selector = FeatureSelector(random_state=config.RANDOM_STATE)
    results_df = selector.run_basic_selection(X_processed, y_processed)
    
    # 可視化の作成
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 特徴量選択手法別のパフォーマンス比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # パフォーマンス比較バープロット
    ax1 = axes[0, 0]
    results_sorted = results_df.sort_values('lgb_cv_mean', ascending=True)
    bars = ax1.barh(range(len(results_sorted)), results_sorted['lgb_cv_mean'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(results_sorted))))
    ax1.set_yticks(range(len(results_sorted)))
    ax1.set_yticklabels(results_sorted['method'])
    ax1.set_xlabel('LightGBM CV AUC')
    ax1.set_title('Feature Selection Methods Performance')
    
    # 値をバーに表示
    for i, (bar, score) in enumerate(zip(bars, results_sorted['lgb_cv_mean'])):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', ha='left', va='center', fontsize=8)
    
    # 特徴量数 vs パフォーマンス
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['n_features'], results_df['lgb_cv_mean'], 
                         c=range(len(results_df)), cmap='viridis', s=100, alpha=0.7)
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('LightGBM CV AUC')
    ax2.set_title('Features Count vs Performance')
    
    # 各点にラベルを追加
    for i, row in results_df.iterrows():
        ax2.annotate(row['method'], (row['n_features'], row['lgb_cv_mean']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # RandomForest vs LightGBM パフォーマンス比較
    ax3 = axes[1, 0]
    ax3.scatter(results_df['rf_cv_mean'], results_df['lgb_cv_mean'], 
               c=range(len(results_df)), cmap='viridis', s=100, alpha=0.7)
    
    # 対角線を描画
    min_score = min(results_df['rf_cv_mean'].min(), results_df['lgb_cv_mean'].min())
    max_score = max(results_df['rf_cv_mean'].max(), results_df['lgb_cv_mean'].max())
    ax3.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5)
    
    ax3.set_xlabel('RandomForest CV AUC')
    ax3.set_ylabel('LightGBM CV AUC')
    ax3.set_title('RandomForest vs LightGBM Performance')
    
    # 手法別の改善率
    ax4 = axes[1, 1]
    baseline_score = results_df[results_df['method'] == 'baseline_all']['lgb_cv_mean'].iloc[0]
    improvement_pct = ((results_df['lgb_cv_mean'] - baseline_score) / baseline_score * 100)
    
    colors = ['red' if x < 0 else 'green' for x in improvement_pct]
    bars = ax4.barh(range(len(results_df)), improvement_pct, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(results_df)))
    ax4.set_yticklabels(results_df['method'])
    ax4.set_xlabel('Improvement over Baseline (%)')
    ax4.set_title('Performance Improvement by Method')
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 値をバーに表示
    for i, (bar, improvement) in enumerate(zip(bars, improvement_pct)):
        ax4.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{improvement:.2f}%', ha='left' if bar.get_width() >= 0 else 'right', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/feature_selection_performance.png', dpi=300, bbox_inches='tight')
    print("Performance comparison plot saved: results/feature_selection_performance.png")
    
    # 2. 特徴量重要度の比較（モデルベース手法）
    if selector.feature_scores_:
        fig, axes = plt.subplots(len(selector.feature_scores_), 1, 
                               figsize=(12, 4 * len(selector.feature_scores_)))
        
        if len(selector.feature_scores_) == 1:
            axes = [axes]
        
        for i, (method, scores_df) in enumerate(selector.feature_scores_.items()):
            ax = axes[i]
            
            # 上位15特徴量を表示
            top_features = scores_df.head(15)
            
            bars = ax.barh(range(len(top_features)), top_features.iloc[:, 1], 
                          color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features.iloc[:, 0])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance: {method}')
            ax.invert_yaxis()
            
            # 値をバーに表示
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("Feature importance comparison plot saved: results/feature_importance_comparison.png")
    
    # 3. 最適特徴量セットの詳細分析
    best_method = results_df.iloc[0]
    best_features = best_method['features']
    
    # 目的変数との相関を計算
    target_corr = X_processed[best_features].corrwith(y_processed).abs().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(target_corr)), target_corr.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(target_corr))))
    plt.yticks(range(len(target_corr)), target_corr.index)
    plt.xlabel('Absolute Correlation with Target')
    plt.title(f'Target Correlation for Best Feature Set: {best_method["method"]}')
    plt.gca().invert_yaxis()
    
    # 値をバーに表示
    for i, (bar, corr) in enumerate(zip(bars, target_corr.values)):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/best_features_target_correlation.png', dpi=300, bbox_inches='tight')
    print("Best features target correlation plot saved: results/best_features_target_correlation.png")
    
    # 4. 結果サマリーテーブルの保存
    summary_data = []
    for _, row in results_df.iterrows():
        summary_data.append({
            'Method': row['method'],
            'Features': len(row['features']),
            'LGB_AUC': f"{row['lgb_cv_mean']:.4f}",
            'RF_AUC': f"{row['rf_cv_mean']:.4f}",
            'Improvement': f"{((row['lgb_cv_mean'] - baseline_score) / baseline_score * 100):+.2f}%",
            'Selected_Features': ', '.join(row['features'][:5]) + ('...' if len(row['features']) > 5 else '')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/feature_selection_summary.csv', index=False)
    print("Feature selection summary saved: results/feature_selection_summary.csv")
    
    # 最良手法の詳細を保存
    with open('results/best_method_details.txt', 'w', encoding='utf-8') as f:
        f.write("=== BEST FEATURE SELECTION METHOD ===\n\n")
        f.write(f"Method: {best_method['method']}\n")
        f.write(f"Number of features: {best_method['n_features']}\n")
        f.write(f"LightGBM CV AUC: {best_method['lgb_cv_mean']:.4f} (+/- {best_method['lgb_cv_std']*2:.4f})\n")
        f.write(f"RandomForest CV AUC: {best_method['rf_cv_mean']:.4f} (+/- {best_method['rf_cv_std']*2:.4f})\n")
        f.write(f"Improvement over baseline: {((best_method['lgb_cv_mean'] - baseline_score) / baseline_score * 100):+.2f}%\n\n")
        f.write("Selected features:\n")
        for i, feature in enumerate(best_features, 1):
            f.write(f"{i:2d}. {feature}\n")
        
        f.write("\nTarget correlation ranking:\n")
        for i, (feature, corr) in enumerate(target_corr.items(), 1):
            f.write(f"{i:2d}. {feature:25s}: {corr:.4f}\n")
    
    print("Best method details saved: results/best_method_details.txt")
    
    return results_df, best_features

if __name__ == "__main__":
    results_df, best_features = create_feature_selection_visualization()
    print(f"\nBest features: {best_features}")
    print(f"Best AUC: {results_df.iloc[0]['lgb_cv_mean']:.4f}")