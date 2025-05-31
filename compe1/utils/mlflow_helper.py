"""
MLflow UI ＆ ngrok helper
Usage:
    from compe1.utils.mlflow_helper import start_mlflow_ui, stop_mlflow_ui
"""
import os, time, subprocess, signal, psutil, warnings
from pathlib import Path

from pyngrok import ngrok, conf

MLFLOW_PORT = 5000

def _get_token() -> str:
    # 1) 環境変数 ＞ 2) .env ＞ 3) エラー
    tok = os.getenv("NGROK_AUTHTOKEN")
    if tok:
        return tok
    # .env 同ディレクトリ探索
    for root in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        env = root / ".env"
        if env.exists():
            from dotenv import load_dotenv
            load_dotenv(env)
            tok = os.getenv("NGROK_AUTHTOKEN")
            if tok:
                return tok
    raise RuntimeError("ngrok token not found. Set NGROK_AUTHTOKEN env or .env file.")

def start_mlflow_ui() -> str:
    """Returns public URL"""
    warnings.filterwarnings("ignore")
    tok = _get_token()
    conf.set_default(conf.PyngrokConfig(auth_token=tok))

    # -- kill existing mlflow on same port --
    for p in psutil.process_iter(["pid","cmdline"]):
        if f"--port {MLFLOW_PORT}" in " ".join(p.info["cmdline"]):
            os.kill(p.info["pid"], signal.SIGTERM)

    # -- mlflow UI subprocess --
    mlflow_proc = subprocess.Popen(
        ["mlflow","ui","--backend-store-uri","file:./mlruns","--host","0.0.0.0","--port", str(MLFLOW_PORT)],
        stdout=open("mlflow.out","w"), stderr=subprocess.STDOUT
    )
    time.sleep(3)

    # -- ngrok tunnel --
    # close previous tunnel on same port
    for t in ngrok.get_tunnels():
        if t.config["addr"].endswith(f":{MLFLOW_PORT}"):
            ngrok.disconnect(t.public_url)
    public_tunnel = ngrok.connect(MLFLOW_PORT, "http")
    print(f"🌐 MLflow UI → {public_tunnel.public_url}")
    return public_tunnel.public_url

def stop_mlflow_ui():
    """close tunnel & kill mlflow proc"""
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)
    for p in psutil.process_iter(["pid","cmdline"]):
        if "mlflow" in " ".join(p.info["cmdline"]):
            try:
                os.kill(p.info["pid"], signal.SIGTERM)
            except Exception:
                pass
    print("🛑 MLflow & ngrok stopped")

# cell: install & helper 呼び出し
!pip -q install mlflow==2.12.1 pyngrok==7.0.0

import os, subprocess, textwrap, time
from pyngrok import ngrok, conf

#--- 環境変数で token を渡す (自PCの ~/.bashrc 等に export 済みなら自動)
token = os.getenv("NGROK_AUTHTOKEN")
if not token:
    raise RuntimeError("環境変数 NGROK_AUTHTOKEN が無い（Colab の Session > 環境変数 で設定）")
conf.set_default(conf.PyngrokConfig(auth_token=token))

#--- mlflow ui をバックグラウンド起動 (stdout をファイルに逃がす)
mlflow_proc = None
public_url = None

def start_mlflow_server_and_ngrok_tunnel():
    global mlflow_proc, public_url
    mlflow_proc = subprocess.Popen(
        ["mlflow","ui","--backend-store-uri","file:./mlruns","--host","0.0.0.0","--port","5000"],
        stdout=open("mlflow.out","w"), stderr=subprocess.STDOUT
    )
    time.sleep(3)                     # 少し待機

    #--- ngrok tunnel
    public_url = ngrok.connect(5000, "http").public_url
    print("🚪  MLflow UI:", public_url)

    #--- keep-alive  (出力が無いと Colab が idle kill する対策)
    def keepalive():
        while True:
            print("…mlflow alive")
            time.sleep(60)
    import threading, atexit, signal
    t = threading.Thread(target=keepalive, daemon=True); t.start()

    def _cleanup(signum=None, frame=None):
        try:
            if public_url:
                ngrok.disconnect(public_url)
        finally:
            if mlflow_proc:
                mlflow_proc.terminate()
        print("🛑 MLflow & ngrok stopped")
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _cleanup)
    atexit.register(_cleanup)

if __name__ == '__main__':
    start_mlflow_server_and_ngrok_tunnel() 