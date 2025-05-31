# --- compe1/utils/mlflow_helper.py --------------------------
import os, subprocess, time, threading, atexit, signal
from pyngrok import ngrok, conf


def start_mlflow_ui(port: int = 5000, backend_uri: str = "file:./mlruns"):
    """
    • mlflow ui をバックグラウンドで立ち上げ  
    • ngrok トンネルを張って外部 URL を返す  
    戻り値: (public_url:str, stop_func:callable)
    """
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError("環境変数 NGROK_AUTHTOKEN が未設定")

    conf.set_default(conf.PyngrokConfig(auth_token=token))

    # --- mlflow UI プロセス
    mlflow_proc = subprocess.Popen(
        ["mlflow", "ui",
         "--backend-store-uri", backend_uri,
         "--host", "0.0.0.0",
         "--port", str(port)],
        stdout=open("mlflow.out", "w"),
        stderr=subprocess.STDOUT
    )
    time.sleep(3)                    # 起動待ち

    # --- ngrok
    public_url = ngrok.connect(port, "http").public_url
    print(f"🚀  MLflow UI: {public_url}")

    # --- keep-alive (Colab idle kill 対策)
    def _keepalive():
        while True:
            print("…mlflow alive")
            time.sleep(60)

    threading.Thread(target=_keepalive, daemon=True).start()

    # --- 停止フック
    def _stop():
        try:
            ngrok.disconnect(public_url)
        finally:
            mlflow_proc.terminate()

    # Colab のセッション終了や Ctrl-C でもクリーンに止める
    atexit.register(_stop)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _stop())

    return public_url, _stop
# ------------------------------------------------------------ 