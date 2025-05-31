"""
MLflow UI ＆ ngrok helper
Usage:
    from compe1.utils.mlflow_helper import start_mlflow_ui, stop_mlflow_ui
"""
import os, time, subprocess, signal, psutil, warnings
from pathlib import Path

from pyngrok import ngrok

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
    ngrok.set_auth_token(tok)

    # -- kill existing mlflow on same port --
    for p in psutil.process_iter(["pid","cmdline"]):
        if f"--port {MLFLOW_PORT}" in " ".join(p.info["cmdline"]):
            os.kill(p.info["pid"], signal.SIGTERM)

    # -- mlflow UI subprocess --
    subprocess.Popen(
        ["mlflow","ui","--backend-store-uri","file:./mlruns",
         "--port", str(MLFLOW_PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
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