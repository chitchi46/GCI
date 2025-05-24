# メイン処理を実行するスクリプト
import pandas as pd
import time
# import matplotlib.pyplot as plt # visualize_feature_vs_target でファイル保存するため、メインでは不要
import os

from src.data_loader import load_train_data, load_test_data, check_data_integrity
from src.eda import summarize_target_distribution, visualize_feature_vs_target # visualize_feature_vs_target をインポート
from src.feature_engineering import create_features, select_features
from src.model import train_lgbm_cv, save_model # model.py の関数 (現状未実装)
# from src.trainer import train_model # trainer.py の関数 (実装済み、model.pyの代替として検討)
from src.tuning import run_optuna_lgbm # (必要に応じて追加)
# from src.ensemble import average_predictions # (必要に応じて追加)
from src.submission import create_submission_file # submission.py の関数 (現状未実装)
# from src.utils import save_submission_file # utils.py の関数 (実装済み、submission.pyの代替として検討)
from src.utils import seed_everything, log_experiment_results, get_git_commit_hash # log_experiment は log_experiment_results を使用

# --- 定数 --- (config.py に移すことも検討)
# config.pyからも読めるようにするなら、そちらに集約し、ここでは import src.config as cfg のようにする
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
TARGET_COLUMN = "Perished"
EXPERIMENT_ID_PREFIX = "exp"
RANDOM_STATE = 42
N_SPLITS_CV = 5
N_TRIALS_OPTUNA = 100 # ユーザー指定は <=100
EDA_PLOTS_DIR = "results/eda_plots" # EDAグラフの保存先ディレクトリ

def main():
    """メイン処理"""
    start_time = time.time()
    current_time_str = time.strftime("%Y%m%d%H%M%S")
    experiment_id = f"{EXPERIMENT_ID_PREFIX}_{current_time_str}"
    git_hash = get_git_commit_hash()
    seed_everything(RANDOM_STATE)
    
    print(f"実験ID: {experiment_id} (Git: {git_hash}) を開始します。")

    # 1. データ読み込みと整合性チェック
    print("\n--- 1. データ読み込みと整合性チェック ---")
    train_df = load_train_data(TRAIN_DATA_PATH)
    test_df = load_test_data(TEST_DATA_PATH)
    train_df = check_data_integrity(train_df, "訓練データ")
    test_df = check_data_integrity(test_df, "テストデータ")

    # 2. EDA
    print("\n--- 2. EDA ---")
    # 目的変数の分布
    summarize_target_distribution(train_df, TARGET_COLUMN)
    
    # 主要な特徴量と目的変数の関係を可視化
    print("\n--- EDA: 主要な特徴量と目的変数の関係の可視化 ---")
    features_to_visualize = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare']
    # Age はビン化も検討 (例: pd.cut(train_df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])) を新たな特徴量として追加し、それを可視化
    # Fare も対数変換やビン化を検討
    
    # EDA_PLOTS_DIR は main.py から見た相対パス
    # visualize_feature_vs_target 内部の output_dir もこのパスを基準とする
    if not os.path.exists(EDA_PLOTS_DIR):
        os.makedirs(EDA_PLOTS_DIR)
        print(f"Created directory for EDA plots: {EDA_PLOTS_DIR}")

    for feature in features_to_visualize:
        if feature in train_df.columns:
            visualize_feature_vs_target(train_df, feature_col=feature, target_col=TARGET_COLUMN, output_dir=EDA_PLOTS_DIR)
        else:
            print(f"Warning: 特徴量 '{feature}' は訓練データに存在しません。スキップします。")

    # (EDAの続き: 相関分析、欠損値処理方針決定、外れ値検討、特徴量エンジニアリングアイデア出しなど)
    # print("\n--- EDA: 相関分析 ---")
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(train_df[features_to_visualize + [TARGET_COLUMN]].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Feature Correlation Heatmap')
    # correlation_save_path = os.path.join(EDA_PLOTS_DIR, "correlation_heatmap.png")
    # plt.savefig(correlation_save_path)
    # plt.close()
    # print(f"Correlation heatmap saved to {correlation_save_path}")

    # 3. 特徴量エンジニアリング
    print("\n--- 3. 特徴量エンジニアリング (スキップ) ---")
    # train_feat_df = create_features(train_df.copy()) # TODO: feature_engineering.py の実装後
    # test_feat_df = create_features(test_df.copy())   # TODO: feature_engineering.py の実装後
    # X_train_selected, selected_feature_names = select_features(train_feat_df, TARGET_COLUMN) # TODO: feature_engineering.py の実装後
    # X_test_selected = test_feat_df[selected_feature_names]
    # y_train = train_df[TARGET_COLUMN]

    # 4. ベースラインモデル学習 (LightGBM)
    print("\n--- 4. ベースラインモデル学習 (スキップ) ---")
    # lgbm_params = { ... } # config.py から読み込むか、ここで定義
    # cv_models, oof_preds, test_preds, cv_score = train_lgbm_cv(X_train_selected, y_train, params=lgbm_params, n_splits=N_SPLITS_CV) # model.py を使う場合
    # cv_models, oof_preds, test_preds, cv_score = train_model(X_train_selected, y_train, X_test_selected, params=lgbm_params, n_splits=N_SPLITS_CV, random_seed=RANDOM_STATE) # trainer.py を使う場合
    # print(f"ベースラインモデル CV Accuracy: {cv_score:.4f}")
    # save_model(cv_models[0], f"baseline_lgbm_{experiment_id}.pkl") # model.py を使う場合 (utils.pyのsave_model_artifactも検討)

    # (以降の処理は現状スキップ)

    end_time = time.time()
    print(f"\n実験ID: {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")
    # log_experiment_results(current_time_str, experiment_id, cv_score if 'cv_score' in locals() else -1, "Initial EDA visualization run", git_hash)

if __name__ == "__main__":
    import os # mainスコープでのosインポートを追加 (EDA_PLOTS_DIR作成用)
    main()
 