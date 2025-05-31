# GCI/compe1/train_lgbm.py
import lightgbm as lgb
import optuna
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from compe1.preprocessing import TitanicPreprocessor # Updated import

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"          # → compe1/data
N_SPLITS = 5
SEED     = 42

def objective(trial, X_df, y):
    params = {
        "boosting_type": trial.suggest_categorical("boost", ["gbdt","dart","goss"]),
        "num_leaves":    trial.suggest_int("num_leaves",  8, 64, step=4),
        "max_depth":     trial.suggest_int("max_depth",   3, 10),
        "learning_rate": trial.suggest_float("lr",       0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child", 10, 40),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
        "lambda_l1":     trial.suggest_float("l1", 0.0, 5.0),
        "lambda_l2":     trial.suggest_float("l2", 0.0, 5.0),
        "n_estimators":  800,
        "objective":     "binary",
        # LightGBM 側の metric は使わず、後で Accuracy を自前評価
        "metric":        "None",
        "random_state":  SEED,
        "verbose":       -1,
        "n_jobs":        -1,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs = []
    for train_idx, valid_idx in cv.split(X_df, y):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        pipe = Pipeline([
            ("prep", TitanicPreprocessor()),
            ("clf",  lgb.LGBMClassifier(**params))
        ])

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        fold_acc = accuracy_score(y_val, preds)
        accs.append(fold_acc)

    # Optuna に渡す値
    mean_acc = sum(accs)/len(accs)
    trial.set_user_attr("cv_accuracy", mean_acc)      # ← 後で取り出せる
    print(f"  ▶︎ CV Accuracy = {mean_acc:.4f}")
    return 1 - mean_acc               # 最小化する値＝ 1-Accuracy

def main():
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
    import warnings, lightgbm as lgbm_warn
    warnings.filterwarnings("ignore", category=FutureWarning)
    study.optimize(lambda t: objective(t, X_df, y), n_trials=50, show_progress_bar=True)

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
    main() 