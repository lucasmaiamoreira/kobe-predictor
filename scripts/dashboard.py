import streamlit as st
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

mlflow.set_tracking_uri("http://localhost:5000")

input_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

model_uri = os.getenv("MLFLOW_MODEL_URI")

loaded_model = mlflow.pyfunc.load_model(model_uri)


def predict(input_data):
    input_data = input_data[input_columns]
    return loaded_model.predict(input_data)

def show_dev_tab():
    st.sidebar.title("Previsão de Arremessos - Ambiente de Desenvolvimento")
    st.title("Previsão com Modelo MLflow - Ambiente de Desenvolvimento")

    dev_data = pd.read_parquet("data/processed/data_filtered_dev.parquet")

    dev_predictions = predict(dev_data)
    
    st.write("Resultado da Previsão - Ambiente de Desenvolvimento:")

    st.write("Dados de Entrada - Ambiente de Desenvolvimento:")
    st.write(dev_data.head())

    st.write("Histograma das Previsões - Ambiente de Desenvolvimento:")
    fig, ax = plt.subplots()
    ax.hist(dev_predictions)
    st.pyplot(fig)

    st.write("Mapa de Calor das Correlações - Ambiente de Desenvolvimento:")
    fig, ax = plt.subplots()
    sns.heatmap(dev_data.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.write("Matriz de Confusão - Ambiente de Desenvolvimento:")
    cm = confusion_matrix(dev_data['shot_made_flag'], dev_predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Previsão')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão')
    st.pyplot(fig)

    st.write("Curva ROC - Ambiente de Desenvolvimento:")
    fpr, tpr, _ = roc_curve(dev_data['shot_made_flag'], dev_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Curva ROC para Ambiente de Desenvolvimento')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    for column in input_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(dev_data[dev_data['shot_made_flag'] == 1][column], kde=True, color='blue', label='Acertou')
        sns.histplot(dev_data[dev_data['shot_made_flag'] == 0][column], kde=True, color='red', label='Errou')
        ax.set_title(f'Histograma da Variável {column} para Arremessos Acertados e Errados')
        ax.set_xlabel(column)
        ax.set_ylabel('Densidade')
        ax.legend()
        st.pyplot(fig)


def show_prod_tab():
    st.sidebar.title("Previsão de Arremessos - Ambiente de Produção")
    st.title("Previsão com Modelo MLflow - Ambiente de Produção")

    prod_data = pd.read_parquet("data/processed/data_filtered_prod.parquet")

    prod_predictions = predict(prod_data)
    
    st.write("Resultado da Previsão - Ambiente de Produção:")

    st.write("Dados de Entrada - Ambiente de Produção:")
    st.write(prod_data.head())

    st.write("Histograma das Previsões - Ambiente de Produção:")
    fig, ax = plt.subplots()
    ax.hist(prod_predictions)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

    st.write("Mapa de Calor das Correlações - Ambiente de Produção:")
    fig, ax = plt.subplots()
    sns.heatmap(prod_data.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.write("Matriz de Confusão - Ambiente de Produção:")
    cm = confusion_matrix(prod_data['shot_made_flag'], prod_predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Previsão')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão')
    st.pyplot(fig)

    st.write("Curva ROC - Ambiente de Produção:")
    fpr, tpr, _ = roc_curve(prod_data['shot_made_flag'], prod_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Curva ROC para Ambiente de Produção')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    for column in input_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(prod_data[prod_data['shot_made_flag'] == 1][column], kde=True, color='blue', label='Acertou')
        sns.histplot(prod_data[prod_data['shot_made_flag'] == 0][column], kde=True, color='red', label='Errou')
        ax.set_title(f'Histograma da Variável {column} para Arremessos Acertados e Errados')
        ax.set_xlabel(column)
        ax.set_ylabel('Densidade')
        ax.legend()
        st.pyplot(fig)


def main():
    selected_tab = st.sidebar.radio("Selecione o Ambiente:", ("Desenvolvimento", "Produção"))

    if selected_tab == "Desenvolvimento":
        show_dev_tab()
    elif selected_tab == "Produção":
        show_prod_tab()

if __name__ == "__main__":
    main()
