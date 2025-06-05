# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

def train_ensemble_models(X, y, X_test, n_splits=5, random_seed=42, n_repeats=3):
    """
    複数のアルゴリズムでアンサンブル学習を行う関数
    """
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
    
    total_folds = n_splits * n_repeats
    
    # 各モデルの設定
    models_config = {
        'lightgbm': lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=random_seed,
            n_estimators=1000
        ),
        'xgboost': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_estimators=1000,
            verbosity=0
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1
        ),
        'logistic_regression': LogisticRegression(
            random_state=random_seed,
            max_iter=1000,
            C=1.0
        )
    }
    
    # 各モデルのOOF予測を格納
    oof_predictions = {}
    test_predictions = {}
    model_scores = {}
    
    for model_name, model in models_config.items():
        print(f"\n--- Training {model_name} ---")
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        cv_scores = []
        
        with mlflow.start_run(run_name=f"ensemble_{model_name}", nested=True):
            # モデルのパラメータをログ
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            for fold, (train_index, val_index) in enumerate(rkf.split(X, y)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                # モデル訓練
                if model_name == 'lightgbm':
                    # Early stopping対応 (LightGBM)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric=['binary_logloss'],  # 明示的に指定
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50, verbose=False)
                        ]
                    )
                elif model_name == 'xgboost':
                    # Early stopping対応 (XGBoost)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # OOF予測
                val_preds_proba = model.predict_proba(X_val)[:, 1]
                oof_preds[val_index] = val_preds_proba
                
                # テスト予測
                test_preds += model.predict_proba(X_test)[:, 1] / total_folds
                
                # スコア計算
                fold_auc = roc_auc_score(y_val, val_preds_proba)
                cv_scores.append(fold_auc)
                
                print(f"  Fold {fold+1}/{total_folds} AUC: {fold_auc:.4f}")
            
            # OOF全体のスコア計算
            oof_auc = roc_auc_score(y, oof_preds)
            oof_logloss = log_loss(y, oof_preds)
            oof_accuracy = accuracy_score(y, (oof_preds > 0.5).astype(int))
            
            mean_cv_auc = np.mean(cv_scores)
            std_cv_auc = np.std(cv_scores)
            
            print(f"  {model_name} OOF AUC: {oof_auc:.4f}")
            print(f"  {model_name} CV AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")
            
            # MLflowにメトリクスを記録
            mlflow.log_metric("oof_auc", oof_auc)
            mlflow.log_metric("oof_logloss", oof_logloss)
            mlflow.log_metric("oof_accuracy", oof_accuracy)
            mlflow.log_metric("cv_auc_mean", mean_cv_auc)
            mlflow.log_metric("cv_auc_std", std_cv_auc)
            
            # 結果を保存
            oof_predictions[model_name] = oof_preds
            test_predictions[model_name] = test_preds
            model_scores[model_name] = {
                'oof_auc': oof_auc,
                'cv_auc_mean': mean_cv_auc,
                'cv_auc_std': std_cv_auc
            }
    
    return oof_predictions, test_predictions, model_scores

def create_ensemble_predictions(oof_predictions, test_predictions, y, ensemble_methods=['average', 'weighted']):
    """
    アンサンブル予測を作成する関数
    """
    ensemble_results = {}
    
    # シンプル平均アンサンブル
    if 'average' in ensemble_methods:
        oof_avg = np.mean(list(oof_predictions.values()), axis=0)
        test_avg = np.mean(list(test_predictions.values()), axis=0)
        
        oof_auc = roc_auc_score(y, oof_avg)
        oof_accuracy = accuracy_score(y, (oof_avg > 0.5).astype(int))
        
        ensemble_results['average'] = {
            'oof_predictions': oof_avg,
            'test_predictions': test_avg,
            'oof_auc': oof_auc,
            'oof_accuracy': oof_accuracy
        }
        
        print(f"Average Ensemble OOF AUC: {oof_auc:.4f}")
    
    # 重み付きアンサンブル (各モデルのOOF AUCを重みとして使用)
    if 'weighted' in ensemble_methods:
        # 各モデルのOOF AUCを計算
        weights = []
        model_names = list(oof_predictions.keys())
        
        for model_name in model_names:
            model_auc = roc_auc_score(y, oof_predictions[model_name])
            weights.append(model_auc)
        
        # 重みを正規化
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        print(f"Model weights: {dict(zip(model_names, weights))}")
        
        # 重み付き平均
        oof_weighted = np.zeros(len(y))
        test_weighted = np.zeros(len(list(test_predictions.values())[0]))
        
        for i, model_name in enumerate(model_names):
            oof_weighted += weights[i] * oof_predictions[model_name]
            test_weighted += weights[i] * test_predictions[model_name]
        
        oof_auc = roc_auc_score(y, oof_weighted)
        oof_accuracy = accuracy_score(y, (oof_weighted > 0.5).astype(int))
        
        ensemble_results['weighted'] = {
            'oof_predictions': oof_weighted,
            'test_predictions': test_weighted,
            'oof_auc': oof_auc,
            'oof_accuracy': oof_accuracy,
            'weights': dict(zip(model_names, weights))
        }
        
        print(f"Weighted Ensemble OOF AUC: {oof_auc:.4f}")
    
    return ensemble_results

def plot_model_comparison(model_scores, save_path=None):
    """
    モデル比較のプロットを作成
    """
    model_names = list(model_scores.keys())
    auc_scores = [model_scores[name]['oof_auc'] for name in model_names]
    cv_means = [model_scores[name]['cv_auc_mean'] for name in model_names]
    cv_stds = [model_scores[name]['cv_auc_std'] for name in model_names]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_scores, width, label='OOF AUC', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, label='CV AUC (mean ± std)', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + cv_stds[i] + 0.005,
                f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    return fig 