"""
Entry-point:  python -m compe1.run_with_mlflow
 1. 起動時に MLflow UI + ngrok トンネル
 2. train_lgbm.main() を呼び出し (autolog でメトリクス保存)
"""
from compe1.utils.mlflow_helper import start_mlflow_ui
import mlflow, mlflow.lightgbm
# import atexit # atexit は mlflow_helper 内部で処理されるため、ここでは不要

# ── 1. UI 起動 ────────────────────────────────────
# start_mlflow_ui は (public_url:str, stop_func:callable) を返す
public_url, stop_mlflow_explicitly = start_mlflow_ui()
print(f"MLflow UI is running at: {public_url}")

# 以前の atexit.register(stop_mlflow_ui) は mlflow_helper 内部の
# atexit.register(_stop) と重複するため削除。
# 明示的に停止したい場合は、最後に stop_mlflow_explicitly() を呼び出す。

# ── 2. autolog ＆ 学習 ───────────────────────────
mlflow.set_experiment("Titanic_LGBM_Optuna")
mlflow.lightgbm.autolog(log_models=False)

from compe1.train_lgbm import main as train_main

if __name__ == "__main__":
    try:
        train_main()
    finally:
        print("Training finished. Stopping MLflow UI and ngrok tunnel...")
        stop_mlflow_explicitly() # 学習後、明示的に停止 