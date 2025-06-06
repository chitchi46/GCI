# -*- coding: utf-8 -*-

# --- 定数定義 ---

# ファイルパス (全てプロジェクトルート My_GCI_Compe1_Project/ からの相対パスとして定義)
TRAIN_DATA_PATH = "compe1/data/train.csv"
TEST_DATA_PATH = "compe1/data/test.csv"
SAMPLE_SUBMISSION_PATH = "compe1/data/sample_submission.csv"

# OUTPUT_DIR, MODEL_DIR, EXP_LOG_PATH は
# スクリプト実行時のカレントワーキングディレクトリからの相対パスを想定しています。
# プロジェクトルート (My_GCI_Compe1_Project) からスクリプトを実行した場合、
# それぞれプロジェクトルート直下の "results/", "models/" を指します。
OUTPUT_DIR = "results/"  # プロジェクトルート/results/ を意図
MODEL_DIR = "models/"    # プロジェクトルート/models/ を意図
EXP_LOG_PATH = "results/exp_log.csv" # プロジェクトルート/results/exp_log.csv を意図

# カラム名
TARGET_COLUMN = "Perished" # 目的変数 (README.ipynb より)
ID_COLUMN = "PassengerId"   # 提出ファイルのIDカラム

# 実験管理
EXPERIMENT_ID_PREFIX = "exp" # 追加

# モデルパラメータ (例)
RANDOM_SEED = 42
RANDOM_STATE = 42 # 実験全体のランダムシード (main.pyでの参照用として追加、RANDOM_SEEDと値を合わせる)

LGB_PARAMS = {
    'objective': 'binary',
    # 'metric': ['binary_logloss', 'accuracy'], # チューニング時に'accuracy'のみとし、trainer側で指定するため削除またはコメントアウト
    'boosting_type': 'gbdt',
    'n_estimators': 850,
    'learning_rate': 0.09059309112098549,
    'num_leaves': 15,
    'max_depth': 5,
    'min_child_samples': 21,
    # 'seed': 42,                 # 固定  <- コメントアウト
    'random_state': RANDOM_STATE, # 追加 (RANDOM_STATE を参照)
    'n_jobs': -1,               # 固定
    'verbose': -1,              # 固定
    'colsample_bytree': 0.5,
    'subsample': 0.7,
    'reg_alpha': 0.005552441584866538,
    'reg_lambda': 0.03482490948808625,
}
N_SPLITS = 5 # CVの分割数
N_SPLITS_CV = 5 # CVの分割数 (main.pyでの参照用として追加、N_SPLITSと値を合わせる)

N_TRIALS_OPTUNA = 30 # Optunaの試行回数を30に変更

# MLflow 設定
DEFAULT_EXPERIMENT_NAME = "Titanic_Survival_Prediction"
MLFLOW_TRACKING_URI = "mlruns" # プロジェクトルート/mlruns を意図
# MLFLOW_EXPERIMENT_NAME = "GCI_Compe1_Titanic" # MLflowの実験名 (指定しない場合は Default)

# --- 前処理関連の設定 ---

# Age補完の設定
AGE_IMPUTATION_METHOD = "random_forest"  # "median", "linear_regression", "random_forest"
AGE_IMPUTATION_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE
}
AGE_CLIP_RANGE = (0, 80)  # 年齢の妥当な範囲

# 外れ値処理の設定
OUTLIER_DETECTION_METHOD = "iqr"  # "iqr", "zscore"
OUTLIER_HANDLING_METHOD = "cap"   # "cap", "winsorize", "remove"
OUTLIER_COLUMNS = ["Fare"]        # 外れ値処理を適用するカラム

# 特徴量エンジニアリングの設定
CREATE_STATISTICAL_FEATURES = True
CREATE_INTERACTION_FEATURES = True

# 年齢グループの設定
AGE_BINS = [0, 12, 18, 30, 50, 80]
AGE_LABELS = ['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']

# 運賃グループの設定
FARE_QUANTILES = 4  # 4分位数でグループ化
FARE_GROUP_LABELS = ['Low', 'Medium', 'High', 'Very_High']

# 家族サイズグループの設定
FAMILY_SIZE_MAPPING = {
    'single': 1,      # 1人
    'small_max': 3,   # 2-3人は小家族
    'large_min': 4    # 4人以上は大家族
}

# タイトルグループの設定
COMMON_TITLES = ['Mr', 'Miss', 'Mrs', 'Master']
TITLE_MAPPING = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", 
    "Mlle": "Miss", "Ms": "Miss", "Sir": "Rare", "Lady": "Rare", 
    "Mme": "Mrs", "Capt": "Rare", "Countess": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", 
    "the Countess": "Rare"
}

# データ品質チェックの設定
VALIDATE_DATA_QUALITY = True
CHECK_DUPLICATES = True
CHECK_INFINITE_VALUES = True

# その他の前処理設定
FARE_LOG_TRANSFORM = True         # Fareの対数変換
EMBARKED_FILL_STRATEGY = "mode"   # Embarkedの欠損値補完戦略
CABIN_EXTRACT_DECK = True         # Cabinからデッキ情報を抽出
CABIN_EXTRACT_NUMBER = True       # Cabinから部屋番号を抽出

# 特徴量エンジニアリング関連の定数
# 例: AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 120]
# 例: FARE_BINS = [-1, 0, 8, 15, 30, 100, 600] # log1p(Fare) を想定

# 前処理関連の定数
# 例: MISSING_AGE_STRATEGY = 'median' # 'mean', 'median', model-based, etc.

# その他
# LOG_LEVEL = "INFO" # loggingのレベル 