# メイン処理を実行するスクリプト
import pandas as pd
import time
# import matplotlib.pyplot as plt # visualize_feature_vs_target でファイル保存するため、メインでは不要
import os
import argparse # コマンドライン引数のため
import optuna
import logging # logging をインポート
import mlflow # MLflow をインポート

from src.data_loader import load_train_data, load_test_data, check_data_integrity
from src.eda import summarize_target_distribution, visualize_feature_vs_target # visualize_feature_vs_target をインポート
from src.preprocessor import preprocess_data # preprocessor をインポート
from src.feature_engineering import create_base_features # feature_engineering から関数をインポート
from src.trainer import train_model # trainer.py の train_model を使用
from src.utils import seed_everything, log_experiment_results, get_git_commit_hash, save_model_artifact, save_submission_file, get_project_root # utilsから必要な関数をインポート (get_project_root追加)
from src import config # config.py をインポート
from src.tuning import run_tuning # tuning.py から run_tuning をインポート
from src.adversarial_validation import run_adversarial_validation # Adversarial Validation 関数をインポート

# compe1 ディレクトリのパスを取得 (main.py が compe1 直下にある前提)
# _COMPE1_ROOT = os.path.abspath(os.path.dirname(__file__)) # 削除
PROJECT_ROOT = get_project_root() # 追加

# --- 定数 --- (config.py に移すことも検討)
# config.pyからも読めるようにするなら、そちらに集約し、ここでは import src.config as cfg のようにする
# TRAIN_DATA_PATH = "data/train.csv" # config.py から読む
# TEST_DATA_PATH = "data/test.csv" # config.py から読む
# TARGET_COLUMN = "Perished" # config.py から読む
# EXPERIMENT_ID_PREFIX = "exp" # config.py に移動済み想定
# RANDOM_STATE = 42 # config.py に移動済み想定
# N_SPLITS_CV = 5 # config.py に移動済み想定
# N_TRIALS_OPTUNA = 100 # ユーザー指定は <=100, config.py に移動済み想定
EDA_PLOTS_DIR_PATH = PROJECT_ROOT / "results" / "eda_plots" # 追加

# OptunaのログレベルをWARNINGに設定
optuna.logging.set_verbosity(optuna.logging.WARNING)

# MLflowの警告レベルを調整
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# logging設定でMLflow関連の警告を抑制
mlflow_logger = logging.getLogger("mlflow")
mlflow_logger.setLevel(logging.ERROR)

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Run the GCI Compe1 pipeline.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna instead of training.")
    parser.add_argument("--adv", "--adversarial_validation", action="store_true", help="Run Adversarial Validation.")
    args = parser.parse_args()

    start_time = time.time()
    current_time_str = time.strftime("%Y%m%d%H%M%S")
    experiment_id = f"{config.EXPERIMENT_ID_PREFIX}_{current_time_str}"
    git_hash = get_git_commit_hash()
    seed_everything(config.RANDOM_STATE)
    
    # MLflow トラッキング URI の設定
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / config.MLFLOW_TRACKING_URI}") # 追加
    
    # Adversarial Validation モードの場合
    if args.adv:
        print("\n--- Adversarial Validation Mode ---")
        overall_auc, feature_importances = run_adversarial_validation()
        if overall_auc is not None:
            print(f"\nAdversarial Validation Overall AUC: {overall_auc:.4f}")
        if feature_importances is not None:
            print("\nAdversarial Validation Top Feature Importances:")
            print(feature_importances.head(20))
        
        end_time = time.time()
        print(f"\nAdversarial Validation finished. Processing time: {end_time - start_time:.2f} seconds")
        return # Adversarial Validation の場合はここで処理を終了
    
    print(f"実験ID: {experiment_id} (Git: {git_hash}) を開始します。")

    # MLflow 実験開始
    with mlflow.start_run(run_name=experiment_id, tags={"git_commit_hash": git_hash}) as run:
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("n_splits_cv", config.N_SPLITS_CV)
        mlflow.log_param("experiment_id_prefix", config.EXPERIMENT_ID_PREFIX)
        # config.py に N_TRIALS_OPTUNA もあるが、チューニング時のみ関連するため、ここではログしない
        # 使用モデルに関する情報もログすると良い (例: mlflow.log_param("model_type", "LightGBM"))

        # 1. データ読み込みと整合性チェック
        print("\n--- 1. データ読み込みと整合性チェック ---")
        train_df = load_train_data(PROJECT_ROOT / config.TRAIN_DATA_PATH) # 修正
        test_df = load_test_data(PROJECT_ROOT / config.TEST_DATA_PATH)   # 修正
        train_df_checked = check_data_integrity(train_df.copy(), "訓練データ") # チェック用コピー
        test_df_checked = check_data_integrity(test_df.copy(), "テストデータ")   # チェック用コピー

        # 2. EDA (前処理前)
        print("\n--- 2. EDA (前処理前) ---")
        summarize_target_distribution(train_df_checked, config.TARGET_COLUMN)
        features_to_visualize_before_preprocessing = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']
        EDA_PLOTS_DIR_PATH.mkdir(parents=True, exist_ok=True) # 修正 (mkdir呼び出しをここに)
        # print(f"Created directory for EDA plots: {EDA_PLOTS_DIR_PATH}") # 以前の修正で追加済みのはずなのでコメントアウトまたは削除
        for feature in features_to_visualize_before_preprocessing:
            if feature in train_df_checked.columns:
                visualize_feature_vs_target(train_df_checked, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=str(EDA_PLOTS_DIR_PATH), filename_prefix="") # 修正
            else:
                print(f"Warning: 特徴量 '{feature}' は訓練データに存在しません。スキップします。")

        # 3. データ前処理
        print("\n--- 3. データ前処理 ---")
        X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy()) 
        
        # 4. EDA (前処理後) - 特徴量エンジニアリング前の状態で確認
        print("\n--- 4. EDA (前処理後) ---")
        processed_train_vis_df_before_fe = X_processed.copy() # FE前のデータで可視化する場合
        processed_train_vis_df_before_fe[config.TARGET_COLUMN] = y_processed.values
        features_to_visualize_after_preprocessing = ['Age', 'Fare']
        for feature in features_to_visualize_after_preprocessing:
            if feature in processed_train_vis_df_before_fe.columns:
                visualize_feature_vs_target(processed_train_vis_df_before_fe, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=str(EDA_PLOTS_DIR_PATH), filename_prefix="processed_") # 修正
            else:
                print(f"Warning: 特徴量 '{feature}' は前処理済み訓練データに存在しません。スキップします。")
            
        # 5. 特徴量エンジニアリング
        print("\n--- 5. 特徴量エンジニアリング ---")
        X_train_final, X_test_final = create_base_features(X_processed, X_test_processed)
        y_train_final = y_processed # yは変更なし

        if args.tune:
            print("\n--- Hyperparameter Tuning Mode ---")
            best_params = run_tuning(X_train_final, y_train_final) # X_train_final と y_train_final を渡す
            print("\n--- Tuning finished. Best parameters found: ---")
            print(best_params)
            print("Tuning process finished. To train with these parameters, re-run without --tune.")
            return # チューニング時はここで処理を終了
        else:
            print("\n--- Training Mode ---")
            # 6. ベースラインモデル学習 (LightGBM using trainer.py)
            print("\n--- 6. ベースラインモデル学習 ---")
            
            # 学習に使用するパラメータをMLflowに記録
            mlflow.log_params(config.LGB_PARAMS)

            cv_models, oof_preds, test_preds, cv_score, cv_auc_score, cv_logloss_score, feature_importance_df = train_model(
                X_train_final, 
                y_train_final, 
                X_test_final, 
                params=config.LGB_PARAMS, # configからLGB_PARAMSを読む
                n_splits=config.N_SPLITS_CV, 
                random_seed=config.RANDOM_STATE,
                n_repeats=3  # RepeatedStratifiedKFoldのrepeats数
            )
            print(f"ベースラインモデル CV Accuracy: {cv_score:.4f}")
            print(f"ベースラインモデル CV AUC: {cv_auc_score:.4f}")
            print(f"ベースラインモデル CV LogLoss: {cv_logloss_score:.4f}")
            
            # メトリクスをMLflowに記録
            mlflow.log_metric("CV_Accuracy_Mean", cv_score)
            mlflow.log_metric("CV_AUC_Mean", cv_auc_score)
            mlflow.log_metric("CV_LogLoss_Mean", cv_logloss_score)
            
            # 特徴量重要度の上位10個を表示
            mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            print(f"\n上位10個の特徴量重要度:")
            for i, (feature, importance) in enumerate(mean_importance.head(10).items()):
                print(f"{i+1:2d}. {feature:20s}: {importance:.4f}")
                
            # MLflowにRepeatedStratifiedKFoldのパラメータを記録
            mlflow.log_param("cv_strategy", "RepeatedStratifiedKFold")
            mlflow.log_param("n_repeats", 3)

        # 7. モデル保存
        print("\n--- 7. モデル保存 ---")
        if cv_models: # 少なくとも1つモデルがあれば保存
            save_model_artifact(cv_models, experiment_id) # utils.pyの関数を使用 (最初のモデルをbaselineとして保存)
        else:
            print("Warning: No models were trained, skipping model saving.")

        # 8. 提出ファイル作成
        print("\n--- 8. 提出ファイル作成 ---")
        # utils.save_submission_file は元の test_df (IDカラムを持つ) と test_preds (確率) を必要とする
        submission_path = save_submission_file(test_df, test_preds, experiment_id) # test_df はID参照のため元のDFを使用
        print(f"Submission file created at: {submission_path}")

        # 9. 実験結果記録
        print("\n--- 9. 実験結果記録 ---")
        log_experiment_results(
            timestamp=current_time_str, 
            exp_id=experiment_id, 
            cv_score=cv_auc_score if 'cv_auc_score' in locals() else -1,  # AUCスコアを主指標に変更
            description="Model with RepeatedStratifiedKFold CV and enhanced feature importance tracking.", # 説明を更新
            git_commit_hash=git_hash
        )

    end_time = time.time()

    print("\n--- 主要な出力先パス ---")
    print(f"実験ID: {experiment_id}")
    if 'submission_path' in locals() and submission_path:
        print(f"提出ファイル: {submission_path}")
    else:
        print("提出ファイル: (生成されませんでした)")
    
    model_dir_path = PROJECT_ROOT / config.MODEL_DIR
    print(f"モデル保存ディレクトリ: {model_dir_path.resolve()}") # .resolve()で絶対パス表示

    exp_log_path_obj = PROJECT_ROOT / config.EXP_LOG_PATH
    print(f"実験ログファイル: {exp_log_path_obj.resolve()}")

    if 'EDA_PLOTS_DIR_PATH' in locals() and EDA_PLOTS_DIR_PATH.exists():
        print(f"EDAプロット保存ディレクトリ: {EDA_PLOTS_DIR_PATH.resolve()}")
    else:
        print("EDAプロット保存ディレクトリ: (生成されませんでした or パス未定義)")

    mlflow_tracking_uri_abs = (PROJECT_ROOT / config.MLFLOW_TRACKING_URI).resolve()
    print(f"MLflow トラッキングURI (絶対パス): file://{mlflow_tracking_uri_abs}")
    print(f"MLflow トラッキングURI (設定値): {mlflow.get_tracking_uri()}")

    # mlflow.end_run() は with ブロックを使ったので不要
    print(f"\n実験ID: {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # mainスコープでのosインポートは不要 (def main()内で完結、または def の中で import)
    main()
 