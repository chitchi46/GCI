# メイン処理を実行するスクリプト
import pandas as pd
import time
# import matplotlib.pyplot as plt # visualize_feature_vs_target でファイル保存するため、メインでは不要
import os

from src.data_loader import load_train_data, load_test_data, check_data_integrity
from src.eda import summarize_target_distribution, visualize_feature_vs_target # visualize_feature_vs_target をインポート
from src.preprocessor import preprocess_data # preprocessor をインポート
from src.feature_engineering import create_features, select_features
from src.model import train_lgbm_cv, save_model # model.py の関数 (現状未実装)
# from src.trainer import train_model # trainer.py の関数 (実装済み、model.pyの代替として検討)
from src.tuning import run_optuna_lgbm # (必要に応じて追加)
# from src.ensemble import average_predictions # (必要に応じて追加)
from src.submission import create_submission_file # submission.py の関数 (現状未実装)
# from src.utils import save_submission_file # utils.py の関数 (実装済み、submission.pyの代替として検討)
from src.utils import seed_everything, log_experiment_results, get_git_commit_hash # log_experiment は log_experiment_results を使用
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
    train_df = load_train_data(config.TRAIN_DATA_PATH)
    test_df = load_test_data(config.TEST_DATA_PATH)
    train_df = check_data_integrity(train_df, "訓練データ")
    test_df = check_data_integrity(test_df, "テストデータ")

    # 2. EDA (前処理前)
    print("\n--- 2. EDA (前処理前) ---")
    # 目的変数の分布
    summarize_target_distribution(train_df, config.TARGET_COLUMN)
    
    # 主要な特徴量と目的変数の関係を可視化
    print("\n--- EDA: 主要な特徴量と目的変数の関係の可視化 (前処理前) ---")
    features_to_visualize_before_preprocessing = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']
    
    # EDA_PLOTS_DIR は main.py から見た相対パス
    # visualize_feature_vs_target 内部の output_dir もこのパスを基準とする
    if not os.path.exists(EDA_PLOTS_DIR):
        os.makedirs(EDA_PLOTS_DIR)
        print(f"Created directory for EDA plots: {EDA_PLOTS_DIR}")

    for feature in features_to_visualize_before_preprocessing:
        if feature in train_df.columns:
            visualize_feature_vs_target(train_df, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=EDA_PLOTS_DIR, filename_prefix="") # 接頭辞なし
        else:
            print(f"Warning: 特徴量 '{feature}' は訓練データに存在しません。スキップします。")

    # 3. データ前処理
    print("\n--- 3. データ前処理 ---")
    # preprocessor.py の preprocess_data を呼び出す際には、train_df と test_df を渡す
    # config は preprocess_data 内部で import src.config されている前提
    X_processed, y_processed, X_test_processed = preprocess_data(train_df.copy(), test_df.copy()) 
    
    # 可視化のために、X_processed にターゲット列を一時的に結合
    # y_processed は Series なので、DataFrame に変換してから結合
    # X_processed は PassengerId を含まない想定（preprocessor.pyの実装による）
    processed_train_vis_df = X_processed.copy()
    processed_train_vis_df[config.TARGET_COLUMN] = y_processed.values # y_processed は preprocess_data から返されるターゲット変数

    # 4. EDA (前処理後)
    print("\n--- 4. EDA (前処理後) ---")
    print("\n--- EDA: 主要な特徴量と目的変数の関係の可視化 (前処理後) ---")
    features_to_visualize_after_preprocessing = ['Age', 'Fare'] # 前処理で変更があったもの

    for feature in features_to_visualize_after_preprocessing:
        if feature in processed_train_vis_df.columns:
            # visualize_feature_vs_target は processed_train_vis_df を受け取る
            visualize_feature_vs_target(processed_train_vis_df, feature_col=feature, target_col=config.TARGET_COLUMN, output_dir=EDA_PLOTS_DIR, filename_prefix="processed_") # 接頭辞 "processed_" を追加
        else:
            print(f"Warning: 特徴量 '{feature}' は前処理済み訓練データに存在しません。スキップします。")
            
    # (EDAの続き: 相関分析、欠損値処理方針決定、外れ値検討、特徴量エンジニアリングアイデア出しなど)
    # print("\n--- EDA: 相関分析 ---")
    # plt.figure(figsize=(12, 10))
    # # 相関を見る際は、数値データのみに絞るのが一般的。X_processed はエンコード済みカテゴリカルも含む。
    # # ここでは、主要な数値特徴量や、前処理後のAge, Fareなど、解釈しやすいものを選ぶと良い。
    # numerical_features_for_corr = ['Age', 'Fare', 'Pclass', 'FamilySize'] # 例
    # features_for_corr = [f for f in numerical_features_for_corr if f in X_processed.columns]
    # if features_for_corr:
    #     correlation_df = X_processed[features_for_corr].copy()
    #     correlation_df[config.TARGET_COLUMN] = y_processed.values
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    #     plt.title('Feature Correlation Heatmap (Processed Data)')
    #     correlation_save_path = os.path.join(EDA_PLOTS_DIR, "processed_correlation_heatmap.png")
    #     try:
    #         plt.savefig(correlation_save_path)
    #         print(f"Correlation heatmap saved to {correlation_save_path}")
    #     except Exception as e:
    #         print(f"Could not save correlation heatmap: {e}")
    #     plt.close()
    # else:
    #     print("No numerical features selected for correlation heatmap.")


    # 5. 特徴量エンジニアリング
    print("\n--- 5. 特徴量エンジニアリング (スキップ) ---")
    # train_feat_df = create_features(X_processed.copy(), y_processed.copy()) # TODO: feature_engineering.py の実装後
    # test_feat_df = create_features(X_test_processed.copy()) # TODO: feature_engineering.py の実装後
    # X_train_selected, selected_feature_names = select_features(train_feat_df, config.TARGET_COLUMN) # TODO: feature_engineering.py の実装後
    # X_test_selected = test_feat_df[selected_feature_names]
    # y_train_final = y_processed # preprocess_data から返されたものをそのまま使う

    # 6. ベースラインモデル学習 (LightGBM)
    print("\n--- 6. ベースラインモデル学習 (スキップ) ---")
    # lgbm_params = config.LGBM_PARAMS # config.py から読み込む
    # cv_models, oof_preds, test_preds, cv_score = train_lgbm_cv(X_train_selected, y_train_final, params=lgbm_params, n_splits=config.N_SPLITS_CV) 
    # print(f"ベースラインモデル CV Accuracy: {cv_score:.4f}")
    # # モデル保存 (utils.py の save_model_artifact を使うなど)
    # if cv_models:
    #    model_save_path = os.path.join(config.MODEL_DIR, f"lgbm_baseline_{experiment_id}.pkl") # MODEL_DIR も config.py で定義
    #    save_model(cv_models[0], model_save_path) # utils.py の save_model_artifact の方が mlflow連携など高機能な場合がある
    #    print(f"Trained model saved to {model_save_path}")

    end_time = time.time()
    print(f"\n実験ID: {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")
    # log_experiment_results(
    #     timestamp=current_time_str, 
    #     experiment_id=experiment_id, 
    #     score=cv_score if 'cv_score' in locals() else -1, 
    #     comment="Ran preprocessing and EDA for Age/Fare after processing.", 
    #     git_hash=git_hash
    # )

if __name__ == "__main__":
    # mainスコープでのosインポートは不要 (def main()内で完結、または def の中で import)
    main()
 