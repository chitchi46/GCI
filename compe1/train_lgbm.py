# GCI/compe1/train_lgbm.py
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocessing import prepare_train_test # Assuming this is in the same directory or PYTHONPATH

DATA_DIR = "data"  # Expected to be relative to the execution directory, e.g., compe1/data
N_SPLITS = 5
SEED     = 42

def objective(trial, X: pd.DataFrame, y: pd.Series):
    params = {
        "objective":        "binary",
        "metric":           "binary_error", # Corresponds to 1.0 - accuracy
        "boosting_type":    trial.suggest_categorical("boost",  ["gbdt","dart","goss"]),
        "num_leaves":       trial.suggest_int("num_leaves",       8, 64, step=4),
        "max_depth":        trial.suggest_int("max_depth",        3, 10),
        "learning_rate":    trial.suggest_float("lr",           0.01, 0.2, log=True),
        "min_child_samples":trial.suggest_int("min_child",       10, 40),
        "subsample":        trial.suggest_float("subsample",    0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample",    0.6, 1.0),
        "lambda_l1":        trial.suggest_float("l1",           0.0, 5.0),
        "lambda_l2":        trial.suggest_float("l2",           0.0, 5.0),
        "seed":             SEED,
        "verbose":          -1, # Suppress LightGBM's own verbose messages
        "n_jobs":           -1,
    }

    # Optuna may pass y as a Series, convert to numpy array for consistent indexing if needed by LGBM
    # However, LGBM Dataset can handle pandas Series directly for labels.
    # X is a DataFrame, y is a Series. Use .iloc for X.
    y_np = y.to_numpy()


    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs = []
    best_iterations_fold = []

    for train_idx, valid_idx in cv.split(X, y_np):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_np[train_idx], y_np[valid_idx]

        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold)

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000, # Max number of boosting rounds
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0) # Suppress messages like [100] valid_0's binary_error: 0.12345
            ]
        )
        
        if gbm.best_iteration is not None and gbm.best_iteration > 0:
            best_iterations_fold.append(gbm.best_iteration)
        else: # Fallback if early stopping didn't occur or best_iteration is not set
            best_iterations_fold.append(2000) 


        preds = (gbm.predict(X_valid_fold, num_iteration=gbm.best_iteration) > 0.5).astype(int)
        accs.append(accuracy_score(y_valid_fold, preds))

    # Store the average best iteration from folds, could be useful
    if best_iterations_fold:
        trial.set_user_attr("best_iteration_avg", int(sum(best_iterations_fold) / len(best_iterations_fold)))

    return 1.0 - (sum(accs) / len(accs))  # Optuna minimizes, so 1 - accuracy

def main():
    # --- Load data ---
    # Assumes train.csv and test.csv are in a 'data' subdirectory 
    # relative to where this script is run.
    # e.g., if script is in /content/compe1/train_lgbm.py, data is in /content/compe1/data/
    df_train_raw = pd.read_csv(f"{DATA_DIR}/train.csv")
    df_test_raw  = pd.read_csv(f"{DATA_DIR}/test.csv")

    # --- Preprocessing ---
    # prepare_train_test is expected to handle feature engineering, imputation, and encoding
    # It should return X_train, y_train, X_test as pandas DataFrames/Series
    X_train, y_train, X_test = prepare_train_test(df_train_raw, df_test_raw)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train columns: {X_train.columns.tolist()}")


    # --- Optuna Hyperparameter Optimization ---
    print("Optuna search ... (can take some time, e.g., 5-7 min for 50 trials)")
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    # Pass X_train (DataFrame) and y_train (Series) to objective
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    # Update with non-tunable params and params derived from study
    best_params.update({
        "objective": "binary",
        "metric": "binary_error",
        "seed": SEED,
        "verbose": -1,
        "n_jobs": -1,
        # Use boosting_type from best_params directly, it's already there
        # "boosting_type": best_params.get("boost", "gbdt") # 'boost' is the trial suggested name
    })
    # Optuna suggests "boost" for "boosting_type", rename if necessary or ensure consistency
    # The `best_params` from Optuna will have `boost` as a key. LightGBM expects `boosting_type`.
    # Let's ensure it's correctly mapped for the final training.
    if 'boost' in best_params:
        best_params['boosting_type'] = best_params.pop('boost')


    print("Best trial achieved score (1.0 - Accuracy):", study.best_value)
    print("Best params found by Optuna:", best_params)

    # Determine the number of boosting rounds for the final model
    # Use the average best_iteration from the best trial, or a fixed number if not available
    num_boost_final = study.best_trial.user_attrs.get("best_iteration_avg", 400) # Fallback to 400
    print(f"Retraining with best params for {num_boost_final} rounds.")

    # --- Re-train on full data with best parameters ---
    lgb_train_full = lgb.Dataset(X_train, label=y_train)
    
    final_gbm = lgb.train(
        best_params,
        lgb_train_full,
        num_boost_round=num_boost_final,
        callbacks=[lgb.log_evaluation(period=0)] # Suppress iteration logs
    )

    # --- Predict on test set ---
    # The comment "Accuracy best thresholdは 0.5 付近で安定" suggests 0.5 is a stable threshold.
    preds_test_proba = final_gbm.predict(X_test, num_iteration=final_gbm.best_iteration if final_gbm.best_iteration else num_boost_final)
    preds_test_binary = (preds_test_proba > 0.5).astype(int)

    # --- Create submission file ---
    submission_df = pd.DataFrame({
        "PassengerId": df_test_raw["PassengerId"], # Use original PassengerId from raw test data
        "Perished": preds_test_binary
    })
    submission_path = "submission_lgbm.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    cv_accuracy = 1.0 - study.best_value
    print(f"Estimated CV Accuracy: {cv_accuracy:.4f}")

if __name__ == "__main__":
    # This allows the script to be run from command line
    # Example: python compe1/train_lgbm.py
    # Ensure that 'data/train.csv' and 'data/test.csv' exist relative to the CWD,
    # and preprocessing.py and feature_engineering.py are importable.
    main() 