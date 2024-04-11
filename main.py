import subprocess
import time
import os
from scripts.preparacao_dados import preparacao_dados
from scripts.treinamento import treinamento
from scripts.aplicacao import aplicacao
import mlflow


def start_mlflow_server():
    # Linux
    subprocess.Popen(["gnome-terminal", "--", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlruns.db"])
    # Windows 
    # subprocess.Popen(["start", "cmd", "/k", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlruns.db"])


def main():
    preparacao_dados()
    treinamento()
    model_uri = aplicacao()

    os.environ["MLFLOW_MODEL_URI"] = model_uri

    subprocess.run(["streamlit", "run", "scripts/dashboard.py"])


if __name__ == "__main__":
    mlflow.end_run()
    start_mlflow_server()
    time.sleep(10)
    main()