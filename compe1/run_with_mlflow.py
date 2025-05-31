"""
Entry-point:  python -m compe1.run_with_mlflow
 1. 起動時に MLflow UI + ngrok トンネル
 2. train_lgbm.main() を呼び出し (autolog でメトリクス保存)
"""
from compe1.utils.mlflow_helper import start_mlflow_ui, stop_mlflow_ui
import mlflow, mlflow.lightgbm, atexit

# ── 1. UI 起動 ────────────────────────────────────
url = start_mlflow_ui()   # prints public URL

# 停止を自動登録
atexit.register(stop_mlflow_ui)

# ── 2. autolog ＆ 学習 ───────────────────────────
mlflow.set_experiment("Titanic_LGBM_Optuna")
mlflow.lightgbm.autolog(log_models=False)

from compe1.train_lgbm import main as train_main

if __name__ == "__main__":
    train_main() 