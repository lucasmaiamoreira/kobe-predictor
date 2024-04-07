import streamlit as st
import pandas as pd
import mlflow

# Função para carregar os dados de produção e as previsões
def load_data_and_predictions():
    # Carregar os dados de produção
    data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")
    # Carregar as previsões
    predictions = pd.read_parquet("output/predictions.parquet")
    return data_prod, predictions

# Função para carregar as métricas do modelo
def load_model_metrics():
    with mlflow.start_run(run_id="f253d7ed10e549199300788109fea8a2") as run:
        # Utilize mlflow.log_metric para registrar as métricas, não mlflow.get_metric
        log_loss_prod = mlflow.log_metric("log_loss_prod")
        f1_score_prod = mlflow.log_metric("f1_score_prod")
    return log_loss_prod, f1_score_prod

# Função principal do dashboard
def main():
    # Título do dashboard
    st.title("Monitoramento do Modelo de Classificação de Kobe")

    # Carregar os dados e as previsões
    data_prod, predictions = load_data_and_predictions()

    # Carregar as métricas do modelo
    log_loss_prod, f1_score_prod = load_model_metrics()

    # Exibir os dados de produção
    st.subheader("Dados de Produção:")
    st.write(data_prod)

    # Exibir as previsões
    st.subheader("Previsões:")
    st.write(predictions)

    # Exibir as métricas do modelo
    st.subheader("Métricas do Modelo:")
    st.write(f"Log Loss: {log_loss_prod}")
    st.write(f"F1 Score: {f1_score_prod}")

if __name__ == "__main__":
    main()
