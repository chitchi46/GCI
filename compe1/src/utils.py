# -*- coding: utf-8 -*-
import os
import pandas as pd
import joblib
import mlflow
import mlflow.lightgbm # save_model_artifact で lightgbm モデルをlogするため
from src import config # config.py から定数をインポート
import random
import numpy as np
import time
import subprocess

RESULTS_DIR = "../results"
LOG_FILE_PATH = os.path.join(RESULTS_DIR, "exp_log.csv")

def seed_everything(seed: int = 42):
    """乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_git_commit_hash() -> str:
    """現在のGitコミットハッシュを取得する関数"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        commit_hash = "N/A"
    return commit_hash

def log_experiment(experiment_id: str, cv_score: float, description: str):
    """実験結果をCSVファイルに記録する関数"""
    # TODO: 実装
    pass

def save_submission_file(test_df, test_preds, exp_id):
    """
    提出ファイルを作成・保存する関数
    output_dir は config から取得
    """
    print("Creating submission file...")
    # test_dfにはID_COLUMNが含まれている想定
    submission_df = pd.DataFrame({config.ID_COLUMN: test_df[config.ID_COLUMN], config.TARGET_COLUMN: (test_preds > 0.5).astype(int)})
    
    output_dir = config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submission_filename = f"submission_{exp_id}.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    mlflow.log_artifact(submission_path)
    return submission_path

def save_model_artifact(models, exp_id, fold_number=None):
    """
    学習済みモデルを保存する関数
    model_dir は config から取得
    """
    print("Saving model artifact(s)...")
    model_dir = config.MODEL_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path_to_log = None # ログする代表パス

    if fold_number is None: # ベースラインモデル (例: 最初のfoldのモデル)
        model_filename = f"baseline.pkl" # 初期指示に合わせる
        model_path = os.path.join(model_dir, model_filename)
        if models: # models リストが空でないことを確認
            joblib.dump(models[0], model_path)
            print(f"Baseline model artifact saved to: {model_path}")
            model_path_to_log = model_path
        else:
            print("No models to save for baseline.pkl")
    else: # CVの場合の個別foldモデル保存（現状、呼び出し側でfold_number指定なし）
        model_filename = f"model_fold{fold_number}_{exp_id}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        if models and len(models) > fold_number -1:
            joblib.dump(models[fold_number-1], model_path)
            print(f"Model artifact for fold {fold_number} saved to: {model_path}")
            # model_path_to_log = model_path # 個別foldもログする場合
        else:
            print(f"Model for fold {fold_number} not found or list too short.")

    # MLflowに物理ファイルを記録 (例: baseline.pkl)
    if model_path_to_log and os.path.exists(model_path_to_log):
        mlflow.log_artifact(model_path_to_log, artifact_path="joblib_models")

    # MLflowにLightGBMモデル自体も記録 (mlflow.lightgbm.autolog() が有効なら通常不要だが明示的に行う場合)
    # if models and mlflow.active_run(): # autologが有効でも重複してログ可能
    #     mlflow.lightgbm.log_model(models[0], artifact_path="lgbm_model_explicit")

    print("Model artifact(s) saving finished.")

def log_experiment_results(timestamp, exp_id, cv_score, description, git_commit_hash="N/A"):
    """
    実験結果をCSVに記録する関数
    exp_log_path は config から取得
    """
    exp_log_path = config.EXP_LOG_PATH
    new_log = pd.DataFrame([{
        "timestamp": timestamp,
        "exp_id": exp_id,
        "CV_score": cv_score,
        "description": description,
        "git_commit_hash": git_commit_hash
    }])
    if os.path.exists(exp_log_path):
        log_df = pd.read_csv(exp_log_path)
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        # ディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(exp_log_path)
        if not os.path.exists(log_dir) and log_dir != '': # log_dirが空文字でないことを確認
            os.makedirs(log_dir)
        log_df = new_log
    log_df.to_csv(exp_log_path, index=False)
    print(f"Experiment results logged to: {exp_log_path}") 