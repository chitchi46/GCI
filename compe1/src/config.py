# -*- coding: utf-8 -*-

# --- 定数定義 ---

# ファイルパス (リポジトリルートからの相対パスを想定)
# main.py が compe1 ディレクトリ直下にある想定なので、そこからの相対パス
TRAIN_DATA_PATH = "compe1/data/train.csv"
TEST_DATA_PATH = "compe1/data/test.csv"
SAMPLE_SUBMISSION_PATH = "compe1/data/sample_submission.csv"
OUTPUT_DIR = "compe1/results/" # main.py から見たパス
MODEL_DIR = "compe1/models/"   # main.py から見たパス
EXP_LOG_PATH = "compe1/results/exp_log.csv" # main.py から見たパス

# カラム名
TARGET_COLUMN = "Perished" # 目的変数 (README.ipynb より)
ID_COLUMN = "PassengerId"   # 提出ファイルのIDカラム

# 実験管理
EXPERIMENT_ID_PREFIX = "exp" # 追加

# モデルパラメータ (例)
LGB_PARAMS = {
    'objective': 'binary',
    # 'metric': ['binary_logloss', 'accuracy'], # チューニング時に'accuracy'のみとし、trainer側で指定するため削除またはコメントアウト
    'boosting_type': 'gbdt',
    'n_estimators': 850,
    'learning_rate': 0.09059309112098549,
    'num_leaves': 15,
    'max_depth': 5,
    'min_child_samples': 21,
    'seed': 42,                 # 固定
    'n_jobs': -1,               # 固定
    'verbose': -1,              # 固定
    'colsample_bytree': 0.5,
    'subsample': 0.7,
    'reg_alpha': 0.005552441584866538,
    'reg_lambda': 0.03482490948808625,
}
N_SPLITS = 5 # CVの分割数
N_SPLITS_CV = 5 # CVの分割数 (main.pyでの参照用として追加、N_SPLITSと値を合わせる)
RANDOM_SEED = 42
RANDOM_STATE = 42 # 実験全体のランダムシード (main.pyでの参照用として追加、RANDOM_SEEDと値を合わせる)

N_TRIALS_OPTUNA = 30 # Optunaの試行回数を30に変更

# MLflow 設定
DEFAULT_EXPERIMENT_NAME = "Titanic_Survival_Prediction" 