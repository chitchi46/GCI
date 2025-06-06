# -*- coding: utf-8 -*-
"""
特徴量選択モジュール
様々な手法を使用して最適な特徴量セットを選択する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 特徴量選択関連
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# 統計関連
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from src import config

class FeatureSelector:
    """包括的な特徴量選択クラス"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.feature_importance_history_ = []
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'f_classif', k: int = 10) -> List[str]:
        """単変量統計による特徴量選択"""
        print(f"\n=== Univariate Feature Selection ({method}) ===")
        
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            # chi2は負の値を扱えないため、最小値を0にシフト
            X_shifted = X - X.min() + 1e-8
            selector = SelectKBest(score_func=chi2, k=k)
            X = X_shifted
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector.fit(X, y)
        
        # スコアを記録
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': getattr(selector, 'pvalues_', [np.nan] * len(X.columns))
        }).sort_values('score', ascending=False)
        
        self.feature_scores_[f'univariate_{method}'] = feature_scores
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features_[f'univariate_{method}'] = selected_features
        
        print(f"Selected {len(selected_features)} features:")
        print(selected_features)
        print(f"\nTop 10 feature scores:")
        print(feature_scores.head(10))
        
        return selected_features
    
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series, 
                           threshold: float = 0.95) -> Tuple[List[str], pd.DataFrame]:
        """相関分析による特徴量選択"""
        print(f"\n=== Correlation Analysis (threshold={threshold}) ===")
        
        # 特徴量間の相関マトリックス
        corr_matrix = X.corr().abs()
        
        # 目的変数との相関
        target_corr = X.corrwith(y).abs().sort_values(ascending=False)
        
        # 高い相関を持つ特徴量ペアを特定
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # 除去する特徴量を決定
        to_drop = []
        high_corr_pairs = []
        
        for column in upper_triangle.columns:
            for index in upper_triangle.index:
                if upper_triangle.loc[index, column] > threshold:
                    high_corr_pairs.append((index, column, upper_triangle.loc[index, column]))
                    # 目的変数との相関が低い方を除去
                    if target_corr[index] > target_corr[column]:
                        if column not in to_drop:
                            to_drop.append(column)
                    else:
                        if index not in to_drop:
                            to_drop.append(index)
        
        # 残す特徴量
        selected_features = [col for col in X.columns if col not in to_drop]
        
        self.selected_features_['correlation_filter'] = selected_features
        
        print(f"High correlation pairs (>{threshold}):")
        for pair in high_corr_pairs[:10]:  # 上位10ペア
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        print(f"\nFeatures to drop: {to_drop}")
        print(f"Selected {len(selected_features)} features after correlation filtering")
        
        # 目的変数との相関
        target_corr_df = pd.DataFrame({
            'feature': target_corr.index,
            'target_correlation': target_corr.values
        })
        
        return selected_features, target_corr_df
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    estimator_type: str = 'rf', n_features: int = 10) -> List[str]:
        """再帰的特徴量除去（RFE）"""
        print(f"\n=== Recursive Feature Elimination ({estimator_type}) ===")
        
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif estimator_type == 'logistic':
            estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        # RFEとRFECVの両方を実行
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        # クロスバリデーション付きRFE
        rfecv = RFECV(
            estimator=estimator, 
            step=1, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        rfecv.fit(X, y)
        
        # 結果を記録
        rfe_features = X.columns[rfe.support_].tolist()
        rfecv_features = X.columns[rfecv.support_].tolist()
        
        self.selected_features_[f'rfe_{estimator_type}'] = rfe_features
        self.selected_features_[f'rfecv_{estimator_type}'] = rfecv_features
        
        print(f"RFE selected features ({len(rfe_features)}): {rfe_features}")
        print(f"RFECV selected features ({len(rfecv_features)}): {rfecv_features}")
        print(f"RFECV optimal number of features: {rfecv.n_features_}")
        
        return rfecv_features  # RFECVの結果を返す
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'lasso') -> List[str]:
        """モデルベースの特徴量選択"""
        print(f"\n=== Model-based Feature Selection ({model_type}) ===")
        
        if model_type == 'lasso':
            # LassoCV で最適な正則化パラメータを選択
            lasso = LassoCV(
                cv=5, 
                random_state=self.random_state,
                max_iter=2000,
                n_jobs=-1
            )
            selector = SelectFromModel(lasso)
            
        elif model_type == 'random_forest':
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            selector = SelectFromModel(rf)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features_[f'model_{model_type}'] = selected_features
        
        # 特徴量重要度を記録
        if hasattr(selector.estimator_, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': selector.estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_scores_[f'model_{model_type}_importance'] = importance_df
            
            print("Feature importance (top 10):")
            print(importance_df.head(10))
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features
    
    def stability_selection(self, X: pd.DataFrame, y: pd.Series, 
                          n_bootstrap: int = 100, threshold: float = 0.6) -> List[str]:
        """安定性選択"""
        print(f"\n=== Stability Selection (n_bootstrap={n_bootstrap}, threshold={threshold}) ===")
        
        n_samples, n_features = X.shape
        selection_counts = np.zeros(n_features)
        
        for i in range(n_bootstrap):
            # ブートストラップサンプリング
            indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # 特徴量の一部をランダム選択
            feature_indices = np.random.choice(
                n_features, 
                size=int(0.8 * n_features), 
                replace=False
            )
            X_boot_features = X_boot.iloc[:, feature_indices]
            
            # Lasso回帰で特徴量選択
            lasso = LassoCV(cv=3, random_state=self.random_state + i, max_iter=1000)
            selector = SelectFromModel(lasso)
            selector.fit(X_boot_features, y_boot)
            
            # 選択された特徴量をカウント
            selected_mask = selector.get_support()
            for j, selected in enumerate(selected_mask):
                if selected:
                    selection_counts[feature_indices[j]] += 1
        
        # 安定性スコアを計算
        stability_scores = selection_counts / n_bootstrap
        stability_df = pd.DataFrame({
            'feature': X.columns,
            'stability_score': stability_scores
        }).sort_values('stability_score', ascending=False)
        
        # 閾値以上の特徴量を選択
        selected_features = stability_df[
            stability_df['stability_score'] >= threshold
        ]['feature'].tolist()
        
        self.feature_scores_['stability_selection'] = stability_df
        self.selected_features_['stability_selection'] = selected_features
        
        print(f"Stability scores (top 10):")
        print(stability_df.head(10))
        print(f"Selected {len(selected_features)} features with stability >= {threshold}")
        
        return selected_features
    
    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series, 
                           features: List[str], method_name: str = "custom") -> Dict:
        """特徴量セットの評価"""
        print(f"\n=== Evaluating Feature Set: {method_name} ===")
        
        X_selected = X[features]
        
        # RandomForestでクロスバリデーション
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(
            rf, X_selected, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # LightGBMでも評価（既存のパラメータを使用）
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(**config.LGB_PARAMS)
            lgb_cv_scores = cross_val_score(
                lgb_model, X_selected, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1
            )
            lgb_mean_score = lgb_cv_scores.mean()
            lgb_std_score = lgb_cv_scores.std()
        except ImportError:
            lgb_mean_score = lgb_std_score = None
        
        results = {
            'method': method_name,
            'n_features': len(features),
            'features': features,
            'rf_cv_mean': cv_scores.mean(),
            'rf_cv_std': cv_scores.std(),
            'lgb_cv_mean': lgb_mean_score,
            'lgb_cv_std': lgb_std_score
        }
        
        print(f"Features ({len(features)}): {features}")
        print(f"RandomForest CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        if lgb_mean_score is not None:
            print(f"LightGBM CV AUC: {lgb_mean_score:.4f} (+/- {lgb_std_score * 2:.4f})")
        
        return results
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    estimator_type: str = 'rf', n_features: int = 10) -> List[str]:
        """再帰的特徴量除去（RFE）"""
        print(f"\n=== Recursive Feature Elimination ({estimator_type}) ===")
        
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif estimator_type == 'logistic':
            estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        # RFEとRFECVの両方を実行
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        # クロスバリデーション付きRFE
        rfecv = RFECV(
            estimator=estimator, 
            step=1, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        rfecv.fit(X, y)
        
        # 結果を記録
        rfe_features = X.columns[rfe.support_].tolist()
        rfecv_features = X.columns[rfecv.support_].tolist()
        
        self.selected_features_[f'rfe_{estimator_type}'] = rfe_features
        self.selected_features_[f'rfecv_{estimator_type}'] = rfecv_features
        
        print(f"RFE selected features ({len(rfe_features)}): {rfe_features}")
        print(f"RFECV selected features ({len(rfecv_features)}): {rfecv_features}")
        print(f"RFECV optimal number of features: {rfecv.n_features_}")
        
        return rfecv_features  # RFECVの結果を返す
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'lasso') -> List[str]:
        """モデルベースの特徴量選択"""
        print(f"\n=== Model-based Feature Selection ({model_type}) ===")
        
        if model_type == 'lasso':
            # LassoCV で最適な正則化パラメータを選択
            lasso = LassoCV(
                cv=5, 
                random_state=self.random_state,
                max_iter=2000,
                n_jobs=-1
            )
            selector = SelectFromModel(lasso)
            
        elif model_type == 'random_forest':
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            selector = SelectFromModel(rf)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features_[f'model_{model_type}'] = selected_features
        
        # 特徴量重要度を記録
        if hasattr(selector.estimator_, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': selector.estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_scores_[f'model_{model_type}_importance'] = importance_df
            
            print("Feature importance (top 10):")
            print(importance_df.head(10))
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features

    def run_basic_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """基本的な特徴量選択の実行"""
        print("=" * 60)
        print("BASIC FEATURE SELECTION")
        print("=" * 60)
        
        print(f"Starting with {len(X.columns)} features: {list(X.columns)}")
        
        evaluation_results = []
        
        # 1. ベースライン（全特徴量）
        baseline_results = self.evaluate_feature_set(X, y, list(X.columns), "baseline_all")
        evaluation_results.append(baseline_results)
        
        # 2. 単変量選択
        for method in ['f_classif', 'mutual_info']:
            for k in [10, 15]:
                if k <= len(X.columns):
                    features = self.univariate_selection(X, y, method=method, k=k)
                    results = self.evaluate_feature_set(X, y, features, f"univariate_{method}_k{k}")
                    evaluation_results.append(results)
        
        # 3. 相関フィルタリング
        corr_features, _ = self.correlation_analysis(X, y, threshold=0.9)
        if len(corr_features) > 0:
            corr_results = self.evaluate_feature_set(X, y, corr_features, "correlation_filter")
            evaluation_results.append(corr_results)
        
        # 4. RFE (RandomForest)
        rfe_features = self.recursive_feature_elimination(X, y, 'rf', 10)
        if len(rfe_features) > 0:
            rfe_results = self.evaluate_feature_set(X, y, rfe_features, "rfe_rf")
            evaluation_results.append(rfe_results)
        
        # 5. モデルベース選択
        for model_type in ['lasso', 'random_forest']:
            features = self.model_based_selection(X, y, model_type)
            if len(features) > 0:
                results = self.evaluate_feature_set(X, y, features, f"model_{model_type}")
                evaluation_results.append(results)
        
        # 結果をデータフレームに整理
        results_df = pd.DataFrame(evaluation_results)
        results_df = results_df.sort_values('lgb_cv_mean', ascending=False)
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION RESULTS SUMMARY")
        print("=" * 60)
        print(results_df[['method', 'n_features', 'rf_cv_mean', 'lgb_cv_mean']].round(4))
        
        return results_df
    
    def plot_feature_importance_comparison(self, output_dir: str = "results"):
        """特徴量重要度の比較プロット"""
        if not self.feature_scores_:
            print("No feature importance data available.")
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        n_methods = len(self.feature_scores_)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, scores_df) in enumerate(self.feature_scores_.items()):
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
        plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance comparison plot saved to {output_dir}/feature_importance_comparison.png")
    
    def get_best_features(self, results_df: pd.DataFrame, top_n: int = 3) -> List[str]:
        """最高性能の特徴量セットを取得"""
        # 上位n個の手法の特徴量を統合
        top_methods = results_df.head(top_n)
        
        feature_counts = {}
        for _, row in top_methods.iterrows():
            for feature in row['features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # 複数の手法で選ばれた特徴量を優先
        best_features = sorted(feature_counts.keys(), 
                             key=lambda x: feature_counts[x], reverse=True)
        
        print(f"\n=== Best Features (from top {top_n} methods) ===")
        print("Feature counts across top methods:")
        for feature in best_features:
            print(f"  {feature}: {feature_counts[feature]}")
        
        return best_features 