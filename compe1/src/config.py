# -*- coding: utf-8 -*-

# --- 定数定義 ---

# ファイルパス (リポジトリルートからの相対パスを想定)
# main.py が compe1 ディレクトリ直下にある想定なので、そこからの相対パス
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
OUTPUT_DIR = "results/" # main.py から見たパス
MODEL_DIR = "models/"   # main.py から見たパス
EXP_LOG_PATH = "results/exp_log.csv" # main.py から見たパス

# カラム名
TARGET_COLUMN = "Perished" # 目的変数 (README.ipynb より)
ID_COLUMN = "PassengerId"   # 提出ファイルのIDカラム

# モデルパラメータ (例)
LGB_PARAMS = {
    'objective': 'binary',
    # 'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
N_SPLITS = 5 # CVの分割数
RANDOM_SEED = 42

# MLflow 設定
DEFAULT_EXPERIMENT_NAME = "Titanic_Survival_Prediction" 