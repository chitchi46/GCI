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
    'metric': ['binary_logloss', 'accuracy'],
    'boosting_type': 'gbdt',
    'n_estimators': 250,
    'learning_rate': 0.05,
    'num_leaves': 25,
    'max_depth': 7,
    'min_child_samples': 30,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
}
N_SPLITS = 5 # CVの分割数
RANDOM_SEED = 42

# MLflow 設定
DEFAULT_EXPERIMENT_NAME = "Titanic_Survival_Prediction" 