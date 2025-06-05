# -*- coding: utf-8 -*-
"""
アンサンブル学習を実行するスクリプト
"""
import pandas as pd
import numpy as np
import time
import mlflow
from pathlib import Path

from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_engineering import create_base_features
from src.ensemble import train_ensemble_models, create_ensemble_predictions, plot_model_comparison
from src.utils import seed_everything, get_project_root, save_submission_file
from src import config

def main():
    """アンサンブル学習のメイン処理"""
    start_time = time.time()
    current_time_str = time.strftime("%Y%m%d%H%M%S")
    experiment_id = f"ensemble_{current_time_str}"
    
    PROJECT_ROOT = get_project_root()
    seed_everything(config.RANDOM_STATE)
    
    # MLflow設定
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / config.MLFLOW_TRACKING_URI}")
    
    print(f"アンサンブル実験ID: {experiment_id} を開始します。")
    
    with mlflow.start_run(run_name=experiment_id, tags={"experiment_type": "ensemble"}) as run:
        # パラメータをログ
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("n_splits_cv", config.N_SPLITS_CV)
        mlflow.log_param("experiment_type", "ensemble")
        
        # 1. データ読み込み
        print("\n--- 1. データ読み込み ---")
        train_df = load_train_data(PROJECT_ROOT / config.TRAIN_DATA_PATH)
        test_df = load_test_data(PROJECT_ROOT / config.TEST_DATA_PATH)
        
        # 2. データ前処理
        print("\n--- 2. データ前処理 ---")
        X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy())
        
        # 3. 特徴量エンジニアリング
        print("\n--- 3. 特徴量エンジニアリング ---")
        X_train_final, X_test_final = create_base_features(X_processed, X_test_processed)
        y_train_final = y_processed
        
        print(f"最終的な特徴量数: {X_train_final.shape[1]}")
        print(f"特徴量: {list(X_train_final.columns)}")
        
        # 4. アンサンブル学習
        print("\n--- 4. アンサンブル学習 ---")
        oof_predictions, test_predictions, model_scores = train_ensemble_models(
            X_train_final, 
            y_train_final, 
            X_test_final,
            n_splits=config.N_SPLITS_CV,
            random_seed=config.RANDOM_STATE,
            n_repeats=3
        )
        
        # 5. モデル比較プロット
        print("\n--- 5. モデル比較プロット作成 ---")
        plot_path = PROJECT_ROOT / "results" / f"model_comparison_{experiment_id}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig = plot_model_comparison(model_scores, str(plot_path))
        mlflow.log_artifact(str(plot_path))
        plt.close(fig)
        
        # 6. アンサンブル予測作成
        print("\n--- 6. アンサンブル予測作成 ---")
        ensemble_results = create_ensemble_predictions(
            oof_predictions, 
            test_predictions, 
            y_train_final,
            ensemble_methods=['average', 'weighted']
        )
        
        # アンサンブル結果をMLflowに記録
        for method, results in ensemble_results.items():
            mlflow.log_metric(f"ensemble_{method}_oof_auc", results['oof_auc'])
            mlflow.log_metric(f"ensemble_{method}_oof_accuracy", results['oof_accuracy'])
            if 'weights' in results:
                for model_name, weight in results['weights'].items():
                    mlflow.log_metric(f"ensemble_weight_{model_name}", weight)
        
        # 7. 最良のアンサンブル手法を選択
        best_method = max(ensemble_results.keys(), 
                         key=lambda x: ensemble_results[x]['oof_auc'])
        best_ensemble = ensemble_results[best_method]
        
        print(f"\n--- 最良のアンサンブル手法: {best_method} ---")
        print(f"OOF AUC: {best_ensemble['oof_auc']:.4f}")
        print(f"OOF Accuracy: {best_ensemble['oof_accuracy']:.4f}")
        
        # 8. 提出ファイル作成
        print("\n--- 7. 提出ファイル作成 ---")
        submission_path = save_submission_file(
            test_df, 
            best_ensemble['test_predictions'], 
            experiment_id
        )
        print(f"Submission file created at: {submission_path}")
        
        # 9. 結果サマリー
        print("\n--- 8. 結果サマリー ---")
        print("個別モデルの性能:")
        for model_name, scores in model_scores.items():
            print(f"  {model_name:20s}: OOF AUC = {scores['oof_auc']:.4f}, CV AUC = {scores['cv_auc_mean']:.4f} (+/- {scores['cv_auc_std']:.4f})")
        
        print(f"\nアンサンブル結果:")
        for method, results in ensemble_results.items():
            print(f"  {method:20s}: OOF AUC = {results['oof_auc']:.4f}, OOF Accuracy = {results['oof_accuracy']:.4f}")
        
        # MLflowに最終結果を記録
        mlflow.log_metric("final_ensemble_auc", best_ensemble['oof_auc'])
        mlflow.log_metric("final_ensemble_accuracy", best_ensemble['oof_accuracy'])
        mlflow.log_param("best_ensemble_method", best_method)
        
    end_time = time.time()
    print(f"\nアンサンブル実験 {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")
    
    return best_ensemble, ensemble_results, model_scores

if __name__ == "__main__":
    # matplotlib必要なのでインポート
    import matplotlib.pyplot as plt
    main() 