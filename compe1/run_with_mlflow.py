"""
Entry-point:  python -m compe1.run_with_mlflow
 1. 起動時に MLflow UI + ngrok トンネル
 2. train_lgbm.main() を呼び出し (autolog でメトリクス保存)
"""
import argparse, time, os
from compe1.utils.mlflow_helper import (
    start_mlflow_server_and_ngrok_tunnel as start_mlflow_ui,
    stop_mlflow_server_and_ngrok_tunnel as stop_mlflow_ui,
)
import mlflow # 汎用 autolog を使うだけなら lightgbm サブモジュールは不要
# import atexit # atexit は mlflow_helper 内部で処理されるため、ここでは不要

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep-ui", action="store_true",
                   help="UI を無制限に残す（セル停止で Kill）")
    p.add_argument("--keep-minutes", type=int, default=0, metavar="M",
                   help="UI を M 分だけ残す（0 → 残さない）")
    return p.parse_args()

# ── 2. autolog ＆ 学習 ───────────────────────────
mlflow.set_experiment("Titanic_LGBM_Optuna")
mlflow.autolog(log_models=True)

if __name__ == "__main__":
    args = parse_args()

    ui_url, *_ = start_mlflow_ui(
        port=5000,
        tracking_uri="file:./mlruns",
        auth_token=os.getenv("NGROK_AUTH_TOKEN")   # ここで環境変数参照
    )

    try:
        # ---- 学習フェーズ ----------------------------------------
        from compe1.train_lgbm import main as train_main
        train_main()

        # ---- UI の保持時間 --------------------------------------
        if args.keep_minutes > 0:
            print(f"Keeping UI for {args.keep_minutes} minutes …")
            time.sleep(args.keep_minutes * 60)
        elif args.keep_ui:
            print("Press STOP (square) in Colab to terminate UI.")
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_mlflow_ui()
        print("Cleaned up MLflow & ngrok.") 