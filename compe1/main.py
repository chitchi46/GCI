# -*- coding: utf-8 -*-
import os
import warnings
import argparse
import datetime
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import joblib

# src.configから定数をインポート
from src import config # もし src が PYTHONPATH に含まれていない場合、相対インポートが必要になることがあります。
                      # Colabで compe1 をカレントにした場合、 from src import config で動作するはず。
from src.data_loader import load_data # load_data をインポート
from src.preprocessor import preprocess_data # preprocess_data をインポート
from src.trainer import train_model # train_model をインポート
from src.utils import save_submission_file, save_model_artifact, log_experiment_results # utilsからインポート

# Matplotlib / Seaborn の設定 (オプション)
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use('ggplot')
# warnings.filterwarnings('ignore') # 警告を非表示にする場合

# --- 定数定義 --- (config.py に移動したため削除)
# TRAIN_DATA_PATH = "data/train.csv"
# TEST_DATA_PATH = "data/test.csv"
# SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
# OUTPUT_DIR = "results/"
# MODEL_DIR = "models/"

# カラム名
# TARGET_COLUMN = "Perished" # 目的変数 (README.ipynb より)
# ID_COLUMN = "PassengerId"   # 提出ファイルのIDカラム

# モデルパラメータ (例)
# LGB_PARAMS = {
#     'objective': 'binary',
#     'metric': 'binary_logloss', # README.ipynb では Accuracy が評価指標だが、LightGBMの学習ではloglossが一般的
#     'boosting_type': 'gbdt',
#     'n_estimators': 10000,
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'max_depth': -1,
#     'seed': 42,
#     'n_jobs': -1,
#     'verbose': -1,
#     'colsample_bytree': 0.8,
#     'subsample': 0.8,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.1,
# }
# N_SPLITS = 5 # CVの分割数
# RANDOM_SEED = 42

# --- 関数定義 --- (utils.py に移動したため削除)
# def save_submission_file(test_df, test_preds, output_dir, exp_id):
#    ...
# def save_model_artifact(models, model_dir, exp_id, fold_number=None):
#    ...
# def log_experiment_results(exp_log_path, timestamp, exp_id, cv_score, description, git_commit_hash="N/A"):
#    ...

def main(args):
    """
    メイン処理を実行する関数
    """
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"Experiment Name: {args.experiment_name}, Run Name: {args.run_name}")
        mlflow.log_param("random_seed", config.RANDOM_SEED)
        mlflow.log_params(config.LGB_PARAMS)

        train_df, test_df = load_data()
        mlflow.log_param("train_data_shape", str(train_df.shape))
        mlflow.log_param("test_data_shape", str(test_df.shape))

        X_train, y_train, X_test = preprocess_data(train_df, test_df.copy())
        
        mlflow.lightgbm.autolog()

        trained_models, oof_predictions, test_predictions, cv_score = train_model(
            X_train, y_train, X_test, config.LGB_PARAMS, config.N_SPLITS, config.RANDOM_SEED
        )
        mlflow.log_metric("mean_cv_accuracy", cv_score)

        # save_model_artifact と save_submission_file の引数を config を使わない形に変更
        save_model_artifact(trained_models, exp_id=args.run_name, fold_number=None)
        submission_file_path = save_submission_file(test_df, test_predictions, exp_id=args.run_name)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception as e:
            print(f"Could not get git hash: {e}")
            git_hash = "N/A"
        
        # log_experiment_results の引数を config を使わない形に変更
        log_experiment_results(
            timestamp,
            args.run_name,
            cv_score,
            f"Baseline LightGBM with {config.N_SPLITS}-fold CV. Features: {X_train.columns.tolist()}",
            git_hash
        )

        print(f"**BEST_CV (Accuracy):** {cv_score:.4f} ▲ from {args.run_name}")
        mlflow.log_metric("final_cv_accuracy", cv_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kaggle competition pipeline.")
    parser.add_argument("--experiment_name", type=str, default=config.DEFAULT_EXPERIMENT_NAME, help="Name of the MLflow experiment.")
    parser.add_argument("--run_name", type=str, default=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Name of the MLflow run.")
    args = parser.parse_args()

    # 警告を一度だけ表示する設定 (オプション)
    # warnings.simplefilter('once', UserWarning)

    # スクリプトのあるディレクトリを基準にパスを解決 (compe1 ディレクトリ直下で実行される想定)
    # os.chdir(os.path.dirname(os.path.abspath(__file__))) # main.py が compe1 の中にある場合

    main(args)

    print("Pipeline finished.") 