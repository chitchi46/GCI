import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
# import sys # 削除
# import os # 削除 (必要なら後で pathlib と共に再導入)

# プロジェクトルートをsys.pathに追加 (単体実行時用)
# このファイルの親の親のディレクトリ (src -> compe1 -> プロジェクトルート)
# _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # 削除
# if _PROJECT_ROOT not in sys.path: # 削除
#     sys.path.insert(0, _PROJECT_ROOT) # 削除

# compe1 ディレクトリを基準パスとする
# _COMPE1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 削除

# from compe1.src import config # 修正: src.config のように変更
# from compe1.src import preprocessor # 修正: src.preprocessor のように変更
# from compe1.src.utils import get_project_root # 修正: src.utils からインポート

from src import config # 変更
from src import preprocessor # 変更
from src.utils import get_project_root # 変更

PROJECT_ROOT = get_project_root() # 追加

def run_adversarial_validation():
    """
    Adversarial Validationを実行する関数
    """
    print("Starting Adversarial Validation...")

    # 1. データの読み込み
    print(f"Loading data from: {PROJECT_ROOT / config.TRAIN_DATA_PATH}, {PROJECT_ROOT / config.TEST_DATA_PATH}") # 修正
    try:
        # train_df_orig = pd.read_csv(os.path.join(_COMPE1_ROOT, config.TRAIN_DATA_PATH)) # 修正
        # test_df_orig = pd.read_csv(os.path.join(_COMPE1_ROOT, config.TEST_DATA_PATH)) # 修正
        train_df_orig = pd.read_csv(PROJECT_ROOT / config.TRAIN_DATA_PATH) # 修正
        test_df_orig = pd.read_csv(PROJECT_ROOT / config.TEST_DATA_PATH)   # 修正
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Make sure paths in config.py are correct relative to project root.")
        print(f"Attempted paths: {PROJECT_ROOT / config.TRAIN_DATA_PATH}, {PROJECT_ROOT / config.TEST_DATA_PATH}") # 修正
        print(f"Original error: {e}")
        return None, None # エラー時はNoneを返す

    print(f"Original train shape: {train_df_orig.shape}, Original test shape: {test_df_orig.shape}")

    # PassengerIdを保持しておく (後で除外するため)
    train_ids = train_df_orig[config.ID_COLUMN] if config.ID_COLUMN in train_df_orig else None
    test_ids = test_df_orig[config.ID_COLUMN] if config.ID_COLUMN in test_df_orig else None

    # 2. 前処理の実行
    # preprocessor.preprocess_data は (X, y, X_test) を返す
    # Adversarial Validationではyは不要
    print("Running preprocessing...")
    # preprocessorはIDカラムを内部で扱うが、最終的に返されるX, X_testからは除外されている想定
    # preprocess_dataがIDカラムを保持している場合があるので、事前に削除するか、返り値を確認する
    # preprocess_data は ID を含む X, X_test を返すことがあるため、ここで明示的に削除
    
    # preprocess_data は (X, y, X_test) を返す。yはここでは使わない。
    # X, X_test にはIDカラムが含まれている可能性があるため、前処理後に除去する。
    X_processed, _, X_test_processed = preprocessor.preprocess_data(train_df_orig.copy(), test_df_orig.copy())
    
    print(f"Processed train shape: {X_processed.shape}, Processed test shape: {X_test_processed.shape}")

    # 3. Adversarial Validation用データセットの作成
    train_adv = X_processed.copy()
    test_adv = X_test_processed.copy()

    # IDカラムが残っていれば除外 (preprocessor.py の返り値に依存)
    if config.ID_COLUMN in train_adv.columns:
        train_adv = train_adv.drop(config.ID_COLUMN, axis=1)
    if config.ID_COLUMN in test_adv.columns:
        test_adv = test_adv.drop(config.ID_COLUMN, axis=1)
        
    # 特徴量の不一致を避けるため、preprocessor後のカラムを信頼する
    # preprocessor.py が返す X と X_test のカラムは一致しているはず

    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    # 念のため、特徴量セットを合わせる (preprocessorが正しく動作していれば不要なはず)
    common_cols = list(set(train_adv.drop('is_test', axis=1).columns) & set(test_adv.drop('is_test', axis=1).columns))
    if len(common_cols) != len(train_adv.drop('is_test', axis=1).columns) or \
       len(common_cols) != len(test_adv.drop('is_test', axis=1).columns):
        print(f"Warning: Feature mismatch after preprocessing. Train features: {train_adv.drop('is_test', axis=1).columns}, Test features: {test_adv.drop('is_test', axis=1).columns}")
        print(f"Using common features: {common_cols}")
        train_adv = train_adv[common_cols + ['is_test']]
        test_adv = test_adv[common_cols + ['is_test']]
    
    adv_df = pd.concat([train_adv, test_adv], ignore_index=True)

    if adv_df.empty or 'is_test' not in adv_df.columns or adv_df.drop('is_test', axis=1).shape[1] == 0:
        print("Error: Adversarial validation dataset is empty, 'is_test' column is missing, or no features found.")
        return None, None

    X_adv = adv_df.drop('is_test', axis=1)
    y_adv = adv_df['is_test']

    print(f"Adversarial dataset shape: X_adv {X_adv.shape}, y_adv {y_adv.shape}")

    # カテゴリカル特徴量のエンコーディング
    # preprocessor.py ですべて数値化されているはずなので、ここでは不要と判断。
    # もしカテゴリ型が残っている場合は、config等からリストを取得して指定する。
    # categorical_features = X_adv.select_dtypes(include=['object', 'category']).columns.tolist()
    # for col in categorical_features:
    #    X_adv[col] = X_adv[col].astype('category')
    # print(f"Categorical features for LGBM: {categorical_features}")
    categorical_features = [] # preprocessor が全て数値に変換している前提

    # 4. LightGBMモデルによる学習と評価
    cv = StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_adv))
    feature_importances = pd.DataFrame()

    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'random_state': config.RANDOM_STATE,
        'n_estimators': 1000, 
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_jobs': -1,
        'verbose': -1,
        # 'colsample_bytree': config.LGB_PARAMS.get('colsample_bytree', 0.8), # configから取得
        # 'subsample': config.LGB_PARAMS.get('subsample', 0.8),
        # 'reg_alpha': config.LGB_PARAMS.get('reg_alpha', 0.0),
        # 'reg_lambda': config.LGB_PARAMS.get('reg_lambda', 0.0),
    }
    # config.LGB_PARAMS の一部を流用（存在すれば）
    # Adversarial Validation 専用のパラメータセットを用意しても良い
    shared_params = ['colsample_bytree', 'subsample', 'reg_alpha', 'reg_lambda', 'max_depth', 'min_child_samples']
    for param in shared_params:
        if param in config.LGB_PARAMS:
            lgbm_params[param] = config.LGB_PARAMS[param]


    for fold, (train_idx, val_idx) in enumerate(cv.split(X_adv, y_adv)):
        X_train_fold, X_val_fold = X_adv.iloc[train_idx], X_adv.iloc[val_idx]
        y_train_fold, y_val_fold = y_adv.iloc[train_idx], y_adv.iloc[val_idx]

        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  # eval_metric='auc', # fitのLGBMClassifierのmetricで指定済み
                  callbacks=[
                      early_stopping(stopping_rounds=100, verbose=-1) # verbose=False or -1
                  ],
                  # categorical_feature=categorical_features if categorical_features else 'auto'
                  # preprocessorで数値化されているはずなので'auto'で問題ないか、空リストを渡す
                  categorical_feature=[] if not categorical_features else categorical_features

        )
        
        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_adv.columns
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold + 1
        feature_importances = pd.concat([feature_importances, fold_importance_df], axis=0)

        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val_fold, val_preds)}")

    overall_auc = roc_auc_score(y_adv, oof_preds)
    print(f"Overall Adversarial Validation AUC: {overall_auc}")

    # 5. 特徴量重要度の表示
    if not feature_importances.empty:
        mean_feature_importances = feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False)
        print("\nTop 20 Feature Importances for Adversarial Validation:")
        print(mean_feature_importances.head(20))
    else:
        print("No feature importances were generated.")
        mean_feature_importances = None


    # TODO: 結果をファイルに保存したり、MLflowにロギングしたりする処理を追加

    print("Adversarial Validation finished.")
    return overall_auc, mean_feature_importances

if __name__ == '__main__':
    # このスクリプトを直接実行した場合の動作
    # (主にデバッグや単体テスト用)
    # 既に sys.path はスクリプト冒頭で設定済み
    
    overall_auc, importances = run_adversarial_validation()
    if overall_auc is not None:
        print(f"\nReturned Overall AUC: {overall_auc}")
    if importances is not None:
        print("\nReturned Feature Importances (Head):")
        print(importances.head()) 