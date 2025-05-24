# メイン処理を実行するスクリプト
import pandas as pd
import time
# import matplotlib.pyplot as plt # visualize_feature_vs_target でファイル保存するため、メインでは不要
import os

from src.data_loader import load_train_data, load_test_data, check_data_integrity
from src.eda import summarize_target_distribution, visualize_feature_vs_target # visualize_feature_vs_target をインポート
from src.preprocessor import preprocess_data # preprocessor をインポート
from src.feature_engineering import create_base_features, create_additional_features # feature_engineering から関数をインポート
from src.trainer import train_model # trainer.py の train_model を使用
from src.utils import seed_everything, log_experiment_results, get_git_commit_hash, save_model_artifact, save_submission_file # utilsから必要な関数をインポート
from src import config # config.py をインポート

# --- 定数 --- (config.py に移すことも検討)
# config.pyからも読めるようにするなら、そちらに集約し、ここでは import src.config as cfg のようにする
# TRAIN_DATA_PATH = "data/train.csv" # config.py から読む
# TEST_DATA_PATH = "data/test.csv" # config.py から読む
# TARGET_COLUMN = "Perished" # config.py から読む
# EXPERIMENT_ID_PREFIX = "exp" # config.py に移動済み想定
# RANDOM_STATE = 42 # config.py に移動済み想定
# N_SPLITS_CV = 5 # config.py に移動済み想定
# N_TRIALS_OPTUNA = 100 # ユーザー指定は <=100, config.py に移動済み想定
EDA_PLOTS_DIR = "results/eda_plots" # EDAグラフの保存先ディレクトリ (config.pyにあっても良い)

def main():
    """メイン処理"""
    start_time = time.time()
    current_time_str = time.strftime("%Y%m%d%H%M%S")
    experiment_id = f"{config.EXPERIMENT_ID_PREFIX}_{current_time_str}"
    git_hash = get_git_commit_hash()
    seed_everything(config.RANDOM_STATE)
    
    print(f"実験ID: {experiment_id} (Git: {git_hash}) を開始します。")

    # 1. データ読み込みと整合性チェック
    print("\n--- 1. データ読み込みと整合性チェック ---")
    train_df = load_train_data(config.TRAIN_DATA_PATH) # 元のtrain_dfを保持（提出ファイル作成時のID参照用）
    test_df = load_test_data(config.TEST_DATA_PATH)   # 元のtest_dfを保持（提出ファイル作成時のID参照用）
    train_df_checked = check_data_integrity(train_df.copy(), "訓練データ") # チェック用コピー
    test_df_checked = check_data_integrity(test_df.copy(), "テストデータ")   # チェック用コピー

    # 2. EDA (前処理前)
    print("\n--- 2. EDA (前処理前) ---")
    summarize_target_distribution(train_df_checked, config.TARGET_COLUMN)
    features_to_visualize_before_preprocessing = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']
    if not os.path.exists(EDA_PLOTS_DIR):
        os.makedirs(EDA_PLOTS_DIR)
        print(f"Created directory for EDA plots: {EDA_PLOTS_DIR}")
    for feature in features_to_visualize_before_preprocessing:
        if feature in train_df_checked.columns:
            visualize_feature_vs_target(train_df_checked, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=EDA_PLOTS_DIR, filename_prefix="")
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
            visualize_feature_vs_target(processed_train_vis_df_before_fe, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=EDA_PLOTS_DIR, filename_prefix="processed_")
        else:
            print(f"Warning: 特徴量 '{feature}' は前処理済み訓練データに存在しません。スキップします。")
            
    # 5. 特徴量エンジニアリング
    print("\n--- 5. 特徴量エンジニアリング ---")
    X_train_base_fe, X_test_base_fe = create_base_features(X_processed, X_test_processed)
    
    # 追加の特徴量エンジニアリング
    X_train_final, X_test_final = create_additional_features(X_train_base_fe, X_test_base_fe) # 新しい関数呼び出し
    y_train_final = y_processed # yは変更なし

    # 必要であれば、特徴量エンジニアリング後の Age_bin, Fare_bin, FamilySize_Category, Ticket_IsNumeric の分布を可視化するコードをここに追加
    # print("\n--- EDA (特徴量エンジニアリング後) ---")
    # fe_train_vis_df = X_train_final.copy()
    # fe_train_vis_df[config.TARGET_COLUMN] = y_train_final.values
    # features_to_visualize_after_fe = ['Age_bin', 'Fare_bin'] 
    # for feature in features_to_visualize_after_fe:
    #     if feature in fe_train_vis_df.columns:
    #         visualize_feature_vs_target(fe_train_vis_df, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=EDA_PLOTS_DIR, filename_prefix="fe_")
    #     else:
    #         print(f"Warning: 特徴量 '{feature}' は特徴量エンジニアリング後の訓練データに存在しません。スキップします。")

    # 6. ベースラインモデル学習 (LightGBM using trainer.py)
    print("\n--- 6. ベースラインモデル学習 ---")
    cv_models, oof_preds, test_preds, cv_score = train_model(
        X_train_final, 
        y_train_final, 
        X_test_final, 
        params=config.LGB_PARAMS, 
        n_splits=config.N_SPLITS_CV, 
        random_seed=config.RANDOM_STATE
    )
    print(f"ベースラインモデル CV Accuracy: {cv_score:.4f}")

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
        cv_score=cv_score if 'cv_score' in locals() else -1, 
        description="Model with binned Age/Fare, FamilySize_Category, Ticket_IsNumeric.", # 説明を更新
        git_commit_hash=git_hash
    )

    end_time = time.time()
    print(f"\n実験ID: {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # mainスコープでのosインポートは不要 (def main()内で完結、または def の中で import)
    main()
 