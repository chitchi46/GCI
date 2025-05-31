# GCI/compe1/train_lgbm.py
import lightgbm as lgb, optuna, pandas as pd, warnings, sys, mlflow
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from compe1.preprocessing import TitanicPreprocessor # Updated import
from compe1.utils.mlflow_helper import start_mlflow_ui
from compe1.src.config import CV_PARAMS # 追加

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"          # → compe1/data
SEED     = 42

# ---- CV split helper ----------------------------------
def make_train_valid_split(X_df, y):
    if CV_PARAMS["type"] == "holdout":
        # stratify には目的変数 y を使用する (もしX_dfのカラムで層化したい場合は別途処理検討)
        return train_test_split(
            X_df, y,
            test_size = CV_PARAMS["test_size"],
            stratify  = y, # CV_PARAMS["stratify_cols"] を使う場合は X_df[CV_PARAMS["stratify_cols"]] だが、多次元になる可能性あり
            random_state = CV_PARAMS["random_state"]
        )
    # KFoldのロジックも必要であればここに追加
    raise ValueError(f"Unsupported CV type: {CV_PARAMS['type']}")

def objective(trial, X_df, y):
    params = {
        "boosting_type": trial.suggest_categorical("boost", ["gbdt","dart","goss"]),
        "num_leaves":    trial.suggest_int("num_leaves", 16, 32, step=4),
        "max_depth":     trial.suggest_int("max_depth",  3,  5),
        "learning_rate": trial.suggest_float("lr",       0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child", 10, 40),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "lambda_l1":     trial.suggest_float("l1", 0.0, 5.0),
        "lambda_l2":     5.0, # 固定値
        "n_estimators":  800,
        "objective":     "binary",
        # LightGBM 側の metric は使わず、後で Accuracy を自前評価
        "metric":        "None",
        "random_state":  SEED,
        "verbose":       -1,
        "n_jobs":        -1,
    }

    # ホールドアウト検証に変更
    X_tr, X_val, y_tr, y_val = make_train_valid_split(X_df, y)

    pipe = Pipeline([
        ("prep", TitanicPreprocessor()),
        ("clf",  lgb.LGBMClassifier(**params))
    ])

    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    acc_score = accuracy_score(y_val, preds)

    # Optuna に渡す値
    trial.set_user_attr("accuracy", acc_score) # ユーザー属性名を変更
    print(f"Trial {trial.number:>2} │ Acc={acc_score:.4f}")
    return 1 - acc_score # 最小化問題なので 1 - accuracy

def main():
    # MLflow UI & ngrok を起動
    mlflow_url, stop_mlflow_hook = start_mlflow_ui(port=5000)
    print("MLflow UI:", mlflow_url)

    # MLflow Experiment を設定
    mlflow.set_experiment("Titanic_LGBM_Optuna")

    # ------ Quiet warnings ------
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    # --- Load data ---
    # Assumes train.csv and test.csv are in a 'data' subdirectory 
    # relative to where this script is run.
    # e.g., if script is in /content/compe1/train_lgbm.py, data is in /content/compe1/data/
    df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
    df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")

    X_df = df_train.drop(["Perished"], axis=1)
    y    = df_train["Perished"].values

    print("Optuna search (leak-free CV)…")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    # mlflow.lightgbm.autolog() は run_with_mlflow.py で設定されている想定
    study.optimize(lambda t: objective(t, X_df, y), n_trials=50, show_progress_bar=False)

    # ------ Summary ------
    best_err = study.best_value
    best_acc = 1 - best_err
    # Optuna trial の user_attrs から精度を取得 (存在しない場合のエラーハンドリングも考慮)
    all_acc  = [t.user_attrs["accuracy"] for t in study.trials if "accuracy" in t.user_attrs]
    mean_acc = sum(all_acc)/len(all_acc) if all_acc else 0 # all_accが空の場合の対処
    print("\n===== Optuna Summary =====")
    print(f"  Best Accuracy : {best_acc:.4f}")
    print(f"  Mean Accuracy : {mean_acc:.4f}")
    print("==========================\n")

    best = {
        **study.best_params,                       # Optuna が探索した値
        "objective": "binary",
        "metric": "None",
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
        "n_estimators": 800                      # 固定値 800
    }
    print("Best params:", best)

    # ------- Final fit on full data -----
    final_pipe = Pipeline([
        ("prep", TitanicPreprocessor()),
        ("clf",  lgb.LGBMClassifier(**best))
    ])
    final_pipe.fit(X_df, y)

    # ------- Predict test ---------------
    X_test_df = df_test.copy() # It's good practice to copy to avoid modifying original test data
    preds = final_pipe.predict(X_test_df)
    sub = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Perished": preds
    })
    sub.to_csv("submission_lgbm.csv", index=False)
    print("submission_lgbm.csv saved.")

if __name__ == "__main__":
    # This allows the script to be run from command line
    # Example: python compe1/train_lgbm.py
    # Ensure that 'data/train.csv' and 'data/test.csv' exist relative to the CWD,
    # and preprocessing.py and feature_engineering.py are importable.
    try:
        main()
    finally:
        if 'stop_mlflow_hook' in locals() and callable(stop_mlflow_hook):
            print("Stopping MLflow UI from train_lgbm.py's finally block...")
            stop_mlflow_hook() # main()が正常/異常終了後も呼ばれるように 