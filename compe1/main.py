# メイン処理を実行するスクリプト
import pandas as pd
import time
import matplotlib.pyplot as plt # グラフ表示のために追加

from src.data_loader import load_train_data, load_test_data, check_data_integrity
from src.eda import summarize_target_distribution
from src.feature_engineering import create_features, select_features
from src.model import train_lgbm_cv, save_model
from src.tuning import run_optuna_lgbm # (必要に応じて追加)
# from src.ensemble import average_predictions # (必要に応じて追加)
from src.submission import create_submission_file
from src.utils import seed_everything, log_experiment, get_git_commit_hash

# --- 定数 --- (config.py に移すことも検討)
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
TARGET_COLUMN = "Perished"
EXPERIMENT_ID_PREFIX = "exp"
RANDOM_STATE = 42
N_SPLITS_CV = 5
N_TRIALS_OPTUNA = 100 # ユーザー指定は <=100

def main():
    """メイン処理"""
    start_time = time.time()
    seed_everything(RANDOM_STATE)
    experiment_id = f"{EXPERIMENT_ID_PREFIX}_{time.strftime('%Y%m%d%H%M%S')}"
    print(f"実験ID: {experiment_id} を開始します。")

    # 1. データ読み込みと整合性チェック
    print("\n--- 1. データ読み込みと整合性チェック ---")
    train_df = load_train_data(TRAIN_DATA_PATH)
    test_df = load_test_data(TEST_DATA_PATH)
    train_df = check_data_integrity(train_df, "訓練データ")
    test_df = check_data_integrity(test_df, "テストデータ")

    # 2. EDA
    print("\n--- 2. EDA ---")
    summarize_target_distribution(train_df, TARGET_COLUMN)
    # plt.show() # summarize_target_distribution で準備されたグラフを表示

    # 3. 特徴量エンジニアリング
    print("\n--- 3. 特徴量エンジニアリング ---")
    # train_feat_df = create_features(train_df.copy()) # TODO: 実装後にコメント解除
    # test_feat_df = create_features(test_df.copy())   # TODO: 実装後にコメント解除
    # selected_features = select_features(train_feat_df, TARGET_COLUMN) # TODO: 実装後にコメント解除
    # X_train = train_feat_df[selected_features]
    # y_train = train_df[TARGET_COLUMN]
    # X_test = test_feat_df[selected_features]
    # test_ids = test_df["PassengerId"] # 提出用にIDを保持

    # 4. ベースラインモデル学習 (LightGBM)
    print("\n--- 4. ベースラインモデル学習 (LightGBM) ---")
    lgbm_params = {
        'objective': 'binary',
        'metric': 'accuracy',
        'random_state': RANDOM_STATE,
        'n_estimators': 1000, # Optunaで調整
        'learning_rate': 0.05, # Optunaで調整
        'num_leaves': 31, # Optunaで調整
        # その他Optunaで調整するパラメータ
        'verbose': -1,
        'n_jobs': -1,
    }
    #oof_preds, test_preds, cv_score, best_model = train_lgbm_cv(X_train, y_train, params=lgbm_params, n_splits=N_SPLITS_CV) # TODO: 実装後にコメント解除
    # print(f"ベースラインモデル (LightGBM) CV Accuracy: {cv_score:.4f}")
    # save_model(best_model, f"baseline_lgbm_{experiment_id}.pkl")

    # 5. Optunaによるチューニング (省略可能、または別スクリプト)
    print("\n--- 5. Optunaによるチューニング (スキップ) ---")
    # study = run_optuna_lgbm(X_train, y_train, n_trials=N_TRIALS_OPTUNA, n_splits=N_SPLITS_CV)
    # print(f"Optuna Best CV Score: {study.best_value:.4f}")
    # print(f"Optuna Best Params: {study.best_params}")
    # tuned_lgbm_params = {**lgbm_params, **study.best_params}
    # oof_preds_tuned, test_preds_tuned, cv_score_tuned, best_model_tuned = train_lgbm_cv(X_train, y_train, params=tuned_lgbm_params, n_splits=N_SPLITS_CV)
    # print(f"チューニング済みLightGBM CV Accuracy: {cv_score_tuned:.4f}")
    # save_model(best_model_tuned, f"tuned_lgbm_{experiment_id}.pkl")

    # 6. アンサンブル (省略可能)
    print("\n--- 6. アンサンブル (スキップ) ---")

    # 7. 提出ファイル作成
    print("\n--- 7. 提出ファイル作成 ---")
    # final_predictions = test_preds_tuned # チューニング後モデルの予測を使用する場合
    # create_submission_file(test_df.copy(), final_predictions, experiment_id, target_col=TARGET_COLUMN) # TODO: 実装後にコメント解除

    # 8. 結果のロギング
    print("\n--- 8. 結果のロギング ---")
    # description = "ベースライン LightGBM モデル"
    # log_experiment(experiment_id, cv_score, description) # TODO: 実装後にコメント解除
    # if 'cv_score_tuned' in locals():
    #     description_tuned = "Optuna チューニング済み LightGBM モデル"
    #     log_experiment(f"{experiment_id}_tuned", cv_score_tuned, description_tuned)

    end_time = time.time()
    print(f"\n実験ID: {experiment_id} が完了しました。処理時間: {end_time - start_time:.2f} 秒")
    # print(f"最良CVスコア: {max(cv_score, cv_score_tuned if 'cv_score_tuned' in locals() else 0):.4f}") # TODO: 修正

if __name__ == "__main__":
    main()
