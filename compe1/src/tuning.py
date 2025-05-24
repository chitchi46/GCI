# Optuna を用いたハイパーパラメータチューニングを担当するモジュール
import optuna
import pandas as pd
from .model import train_lgbm_cv # model.py から関数をインポート
from sklearn.model_selection import StratifiedKFold # CVのために追加

# src.trainer や src.config をインポートする必要がある
from src import config
from src.trainer import train_model # 既存のtrain_modelを再利用するか、ここで専用の学習ロジックを持つか検討

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna の目的関数"""
    # --- ハイパーパラメータの定義 ---
    # LGB_PARAMS からベースのパラメータを取得し、trialで提案された値で上書き
    params = config.LGB_PARAMS.copy() # dict.copy() を使う
    params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000, step=50)
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
    params['num_leaves'] = trial.suggest_int('num_leaves', 15, 63) # 2^max_depth より小さいことが推奨される場合もある
    params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 50)
    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0, step=0.1)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1)
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True)
    
    # metric は accuracy に固定（あるいはconfigから読むようにする）
    params['metric'] = 'accuracy' 
    # verbose も -1 に固定、seed も固定
    params['verbose'] = -1
    params['seed'] = config.RANDOM_SEED

    # --- モデル学習と評価 ---
    # 既存の train_model を利用する場合。train_model がCVスコアを返す必要がある。
    # train_model は現在、モデルのリスト、oof予測、test予測、CVスコアのタプルを返す。
    # ここではCVスコアのみが必要。
    try:
        _, _, _, cv_score = train_model(
            X_train=X,
            y_train=y,
            X_test=pd.DataFrame(), # テストデータはチューニング時には不要なので空のDFを渡す
            params=params,
            n_splits=config.N_SPLITS_CV,
            random_seed=config.RANDOM_STATE
        )
        return cv_score # Accuracy を最大化する
    except Exception as e:
        print(f"An error occurred during trial: {e}")
        # エラーが発生した場合、この試行を失敗として扱い、Optunaに低い値を返す
        # (あるいは optuna.exceptions.TrialPruned() をraiseする)
        return -1.0 # または適切なエラー値

def run_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Optuna を使ったハイパーパラメータチューニングを実行する"""
    print(f"Starting hyperparameter tuning with Optuna ({config.N_TRIALS_OPTUNA} trials)...")
    
    # functools.partial を使って objective に追加の引数 (X_train, y_train) を渡す
    # from functools import partial
    # objective_with_data = partial(objective, X=X_train, y=y_train)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # lambda を使って objective に引数を渡す方がシンプル
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=config.N_TRIALS_OPTUNA)

    print("\nHyperparameter tuning finished.")
    print(f"Best trial - Value (CV Accuracy): {study.best_value:.4f}")
    print("Best trial - Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # TODO: 最適なパラメータをファイルに保存する処理 (例: JSON)
    # best_params_path = "config/best_lgb_params.json"
    # import json
    # with open(best_params_path, 'w') as f:
    #     json.dump(study.best_params, f, indent=4)
    # print(f"Best parameters saved to {best_params_path}")

    return study.best_params

if __name__ == '__main__':
    # このスクリプトを直接実行した場合のテスト用コード
    # 実際のデータ読み込みや前処理は main.py で行われる想定
    print("Running a dummy tuning process...")
    
    # ダミーのデータを作成
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
    y_dummy_series = pd.Series(y_dummy)

    # 既存の preprocessor や feature_engineering を通した後のデータに近い形を想定
    # 実際には main.py から X_train_final, y_train_final を渡す
    
    # config.LGB_PARAMS の 'metric' をチューニング用に調整
    original_metric = config.LGB_PARAMS.get('metric')
    config.LGB_PARAMS['metric'] = 'accuracy' # objective内でaccuracyを使うため

    best_params = run_tuning(X_dummy_df, y_dummy_series)
    
    # 'metric' を元に戻す
    if original_metric is not None:
        config.LGB_PARAMS['metric'] = original_metric
    else:
        del config.LGB_PARAMS['metric'] # もし元々存在しなかった場合

    print("\nDummy tuning process finished.")
    print("Best parameters found:")
    print(best_params)

def objective_lgbm(trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> float:
    """LightGBM の Optuna 最適化関数"""
    # TODO: 実装
    pass

def run_optuna_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 100, n_splits: int = 5, random_state: int = 42) -> optuna.study.Study:
    """LightGBM の Optuna チューニングを実行する関数"""
    # TODO: 実装
    pass 