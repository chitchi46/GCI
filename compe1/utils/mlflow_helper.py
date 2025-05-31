# --- compe1/utils/mlflow_helper.py --------------------------
import os, signal, subprocess, atexit, time, re
from pathlib import Path
from typing import Tuple, Optional

_MLFLOW_PROC: Optional[subprocess.Popen] = None
_NGROK_PROC:  Optional[subprocess.Popen] = None
_UI_URL:      Optional[str]              = None

def _kill(proc: Optional[subprocess.Popen], name: str):
    """Kill a subprocess and its process-group (Linux / Colab)."""
    if proc and proc.poll() is None:                       # still alive?
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            print(f"[WARN] kill {name}: {e}")

def _cleanup():
    """atexit フック：ノートブック終了・強制 Reset 時にも呼ばれる。"""
    _kill(_NGROK_PROC,  "ngrok")
    _kill(_MLFLOW_PROC, "mlflow")

atexit.register(_cleanup)      # <<<<<< ① 最終保険

# ------------------------------------------------------------------
def start_mlflow_server_and_ngrok_tunnel(
        port: int = 5000,
        tracking_uri: str = "file:./mlruns",
        auth_token: str | None = None,
) -> Tuple[str, subprocess.Popen, subprocess.Popen]:
    """
    返り値:
        ui_url, mlflow_proc, ngrok_proc
    """
    global _MLFLOW_PROC, _NGROK_PROC, _UI_URL

    # --- 1) MLflow サーバをサブプロセスで起動 ---------------------
    mlflow_cmd = [
        "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    _MLFLOW_PROC = subprocess.Popen(
        mlflow_cmd,
        preexec_fn=os.setsid,              # 子 + その孫も一括 kill できるよう新しい PG
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    time.sleep(2)                          # ポートが開くまで軽く待機

    # --- 2) ngrok トンネル ----------------------------------------
    if auth_token:
        subprocess.run(["ngrok", "config", "add-authtoken", auth_token],
                       check=False, stdout=subprocess.DEVNULL)

    _NGROK_PROC = subprocess.Popen(
        ["ngrok", "http", str(port)],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # ngrok が URL を吐くまでログを少し読む
    _UI_URL = None # ループ前に初期化
    for _ in range(120):                    # ≒ 12 秒待つ
        line = ""
        if _NGROK_PROC.stdout:
            line = _NGROK_PROC.stdout.readline().strip() # text=Trueなら既にstr, decode不要
        
        print(f"[DEBUG ngrok line] {line}") # デバッグ出力
        
        # 現在のURLパースロジック (より堅牢な正規表現に変更する)
        # if "url=" in line:
        #    for part in line.split():
        #        if part.startswith("url=") and part.endswith(".app"):
        #            _UI_URL = part.split("=", 1)[1].replace("http://", "https://")
        #            break
        m = re.search(r"url=(https://[0-9a-z\-]+\.ngrok-free\.app)", line) # ngrok-free.app に対応した正規表現
        if m:
            _UI_URL = m.group(1) # キャプチャグループ1 (https://...) を取得
            break

        if not line and _NGROK_PROC.poll() is not None:
            print("[WARN] ngrok process may have exited prematurely or stdout is closed.")
            break
        time.sleep(0.1)              # 待機

    print(f"MLflow UI: {_UI_URL or '(URL not found)'}")

    return _UI_URL, _MLFLOW_PROC, _NGROK_PROC

# ------------------------------------------------------------------
def stop_mlflow_server_and_ngrok_tunnel():
    """手動クリーンアップ（try/finally から呼び出し）"""
    _kill(_NGROK_PROC,  "ngrok")
    _kill(_MLFLOW_PROC, "mlflow")
# ------------------------------------------------------------------ 