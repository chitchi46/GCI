# --- compe1/utils/mlflow_helper.py --------------------------
import os, signal, subprocess, atexit, time, re
from pathlib import Path
from typing import Tuple, Optional
import sys # sysモジュールをインポート

_MLFLOW_PROC: Optional[subprocess.Popen] = None
_NGROK_PROC:  Optional[subprocess.Popen] = None           # subprocess.Popen オブジェクト
_UI_URL:      Optional[str]              = None           # 取得できた URL 文字列

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

    # ngrokの出力からURLを抽出するための正規表現パターン。
    # 例: "url=https://xxxxxxxx.ngrok-free.app"
    # ユーザーはURLそのもののパターンを提示したが、ngrokのログ形式 "url=XXX" を考慮する。
    # 既存の正規表現 `r"url=(https://[0-9a-z\-]\+\.ngrok-free\.app)"` は `ngrok-free.app` のみだったが、
    # ユーザー提示の `r"https://[0-9a-z\-]\+\.ngrok(-free)?\.app"` は `ngrok.app` と `ngrok-free.app` の両方に対応する。
    # これを組み合わせる。
    url_extract_pat = re.compile(r"url=(https://[0-9a-z\-]+\.ngrok(?:-free)?\.app)")

    deadline = time.time() + 20 # timeout 20秒

    log_buffer = [] # デバッグ用にログを一時的に保存

    while time.time() < deadline:
        if not _NGROK_PROC.stdout: # stdoutがNoneの場合は読み取り不可
            print("[WARN] ngrok stdout is not available.", file=sys.stderr)
            break
            
        line = _NGROK_PROC.stdout.readline() # text=True なので str
        if not line: # readlineがEOFまたは空行を返した場合
            if _NGROK_PROC.poll() is not None: # プロセスが終了したか確認
                print("[WARN] ngrok process exited prematurely while reading stdout.", file=sys.stderr)
                break
            time.sleep(0.1) # プロセスは生きているが新しい行がない場合、少し待つ
            continue
        
        line = line.strip()
        print("[ngrok]", line, file=sys.stderr)
        
        m = url_extract_pat.search(line)
        if m:
            _UI_URL = m.group(1) # キャプチャグループ1 (https://...) を取得
            break
        log_buffer.append(line) # URLが見つからない場合、ログバッファに追加
    
    if _UI_URL is None:
        # stdout ダンプ済みなので原因はログに残る
        # 既存の警告メッセージは削除し、RuntimeErrorを発生させる
        # print("[WARN] Could not find ngrok URL within timeout. Reviewing log buffer:", file=sys.stderr)
        # for i, log_line in enumerate(log_buffer):
        #     print(f"[ngrok log buffer {i+1}/{len(log_buffer)}] {log_line}", file=sys.stderr)
        # if _NGROK_PROC.poll() is not None:
        #      print("[WARN] ngrok process also exited.", file=sys.stderr)
        raise RuntimeError("✖ ngrok URL を取得できませんでした。上の [ngrok] 行を確認してください。")

    # MLflow UIのURLを新しいフォーマットで出力
    print(f"MLflow UI ➜ {_UI_URL}")
    # print(f"MLflow UI: {_UI_URL or '(URL not found)'}") # 古いprint文は削除

    return _UI_URL, _MLFLOW_PROC, _NGROK_PROC

# ------------------------------------------------------------------
def stop_mlflow_server_and_ngrok_tunnel():
    """手動クリーンアップ（try/finally から呼び出し）"""
    _kill(_NGROK_PROC,  "ngrok")
    _kill(_MLFLOW_PROC, "mlflow")
# ------------------------------------------------------------------
