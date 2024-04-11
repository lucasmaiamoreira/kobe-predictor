import subprocess
from scripts.preparacao_dados import preparacao_dados
from scripts.treinamento import treinamento
from scripts.aplicacao import aplicacao
import os

def main():
    preparacao_dados()
    treinamento()
    model_uri = aplicacao()
    print(model_uri)

    os.environ["MLFLOW_MODEL_URI"] = model_uri

    subprocess.run(["streamlit", "run", "scripts/dashboard.py"])


if __name__ == "__main__":
    main()
