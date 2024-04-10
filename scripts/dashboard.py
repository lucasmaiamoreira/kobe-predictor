import streamlit as st
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")

# Carregar o modelo MLflow
model_uri = "mlflow-artifacts:/0/84103a8bab7c4132b091fac101c1508b/artifacts/final_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Função para fazer previsões com o modelo
def predict(input_data):
    return loaded_model.predict(input_data)

# Função principal do aplicativo Streamlit
def main():
    # Interface do usuário para entrada de dados
    st.title("Previsão com Modelo MLflow")

    st.sidebar.title("Previsão de Arremessos")

    # Adicione sliders para entrada de dados
    lat = st.sidebar.slider("Latitude", min_value=-90.0, max_value=90.0, step=0.1, value=0.0)
    lon = st.sidebar.slider("Longitude", min_value=-180.0, max_value=180.0, step=0.1, value=0.0)
    minutes_remaining = st.sidebar.slider("Minutos Restantes", min_value=0, max_value=48, step=1, value=0)
    period = st.sidebar.slider("Período", min_value=1, max_value=4, step=1, value=1)
    playoffs = st.sidebar.radio("Playoffs", options=["Sim", "Não"], index=0)
    playoffs = 1 if playoffs == "Sim" else 0
    shot_distance = st.sidebar.slider("Distância do Arremesso", min_value=0, max_value=50, step=1, value=0)


    if st.sidebar.button("Prever"):
        # Organize os dados de entrada em uma lista
        input_data = [lat, lon, minutes_remaining, period, playoffs, shot_distance]

        # Fazer previsões com o modelo
        prediction = predict([input_data])

        # Exibir resultado da previsão
        st.write("Resultado da Previsão:", prediction)

        # Visualização dos dados de entrada
        input_df = pd.DataFrame({'Latitude': [lat], 'Longitude': [lon], 'Minutos Restantes': [minutes_remaining],
                                 'Período': [period], 'Playoffs': [playoffs], 'Distância do Arremesso': [shot_distance]})
        st.write("Dados de Entrada:")
        st.write(input_df)

        # Histograma das previsões
        st.write("Histograma das Previsões:")
        fig, ax = plt.subplots()
        ax.hist(prediction)
        st.pyplot(fig)

        # Mapa de calor das correlações entre as características
        st.write("Mapa de Calor das Correlações:")
        fig, ax = plt.subplots()
        sns.heatmap(input_df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()

