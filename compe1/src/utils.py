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
from pathlib import Path

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
    output_dir は config から取得し、プロジェクトルート基準で解決
    """
    PROJECT_ROOT = get_project_root()
    print("Creating submission file...")
    submission_df = pd.DataFrame({config.ID_COLUMN: test_df[config.ID_COLUMN], config.TARGET_COLUMN: (test_preds > 0.5).astype(int)})
    
    output_dir_path = PROJECT_ROOT / config.OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)

    submission_filename = f"submission_{exp_id}.csv"
    submission_path = output_dir_path / submission_filename
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    mlflow.log_artifact(str(submission_path))
    return str(submission_path)

def save_model_artifact(models, exp_id, fold_number=None):
    """
    学習済みモデルを保存する関数
    model_dir は config から取得し、プロジェクトルート基準で解決
    """
    PROJECT_ROOT = get_project_root()
    print("Saving model artifact(s)...")
    model_dir_path = PROJECT_ROOT / config.MODEL_DIR
    model_dir_path.mkdir(parents=True, exist_ok=True)

    model_path_to_log_str = None # ログする代表パス (文字列)

    if fold_number is None: 
        model_filename = "baseline.pkl" 
        model_path = model_dir_path / model_filename
        if models: 
            joblib.dump(models[0], model_path)
            print(f"Baseline model artifact saved to: {model_path}")
            model_path_to_log_str = str(model_path)
        else:
            print("No models to save for baseline.pkl")
    else: 
        model_filename = f"model_fold{fold_number}_{exp_id}.pkl"
        model_path = model_dir_path / model_filename
        if models and len(models) > fold_number -1:
            joblib.dump(models[fold_number-1], model_path)
            print(f"Model artifact for fold {fold_number} saved to: {model_path}")
            # model_path_to_log_str = str(model_path) # 個別foldもログする場合
        else:
            print(f"Model for fold {fold_number} not found or list too short.")

    if model_path_to_log_str and Path(model_path_to_log_str).exists():
        mlflow.log_artifact(model_path_to_log_str, artifact_path="joblib_models")

    print("Model artifact(s) saving finished.")

def log_experiment_results(timestamp, exp_id, cv_score, description, git_commit_hash="N/A"):
    """
    実験結果をCSVに記録する関数
    exp_log_path は config から取得し、プロジェクトルート基準で解決
    """
    PROJECT_ROOT = get_project_root()
    exp_log_path_obj = PROJECT_ROOT / config.EXP_LOG_PATH
    
    new_log = pd.DataFrame([{
        "timestamp": timestamp,
        "exp_id": exp_id,
        "CV_score": cv_score,
        "description": description,
        "git_commit_hash": git_commit_hash
    }])
    
    log_dir_path = exp_log_path_obj.parent
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if exp_log_path_obj.exists():
        log_df = pd.read_csv(exp_log_path_obj)
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        log_df = new_log
    log_df.to_csv(exp_log_path_obj, index=False)
    print(f"Experiment results logged to: {exp_log_path_obj}")

def get_project_root() -> Path:
    # utils.py is in My_GCI_Compe1_Project/compe1/src/
    # We need to go up three levels to reach My_GCI_Compe1_Project/
    return Path(__file__).resolve().parent.parent.parent 