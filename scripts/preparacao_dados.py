# import pandas as pd
# from sklearn.model_selection import train_test_split
# import mlflow
# import mlflow.sklearn

# # Iniciar uma nova run do MLflow com o nome "PreparacaoDados"
# mlflow.start_run(run_name="PreparacaoDados")

# # Carregar os dados de treinamento e produção
# data_dev = pd.read_parquet("data/dataset_kobe_dev.parquet")
# data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

# # Selecionar as colunas relevantes
# relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
# data_dev_filtered = data_dev[relevant_columns].dropna()
# data_prod_filtered = data_prod[relevant_columns[:-1]].dropna()  # Excluir a coluna 'shot_made_flag' dos dados de produção

# # Salvar o dataset processado
# data_dev_filtered.to_parquet("data/processed/data_filtered_dev.parquet")
# data_prod_filtered.to_parquet("data/processed/data_filtered_prod.parquet")

# # Separar os dados em treino e teste (80% treino, 20% teste)
# X_train, X_test, y_train, y_test = train_test_split(data_dev_filtered.drop(columns=['shot_made_flag']), 
#                                                     data_dev_filtered['shot_made_flag'], 
#                                                     test_size=0.2, 
#                                                     stratify=data_dev_filtered['shot_made_flag'], 
#                                                     random_state=42)

# # Incluir a coluna 'shot_made_flag' nos datasets de treino e teste
# train_data = pd.concat([X_train, y_train], axis=1)
# test_data = pd.concat([X_test, y_test], axis=1)

# # Salvar os datasets de treino e teste em arquivos Parquet
# train_data.to_parquet("data/processed/base_train.parquet")
# test_data.to_parquet("data/processed/base_test.parquet")

# # Registrar os parâmetros e métricas no MLflow
# mlflow.log_param("percent_test", 0.2)
# mlflow.log_metric("train_size", len(X_train))
# mlflow.log_metric("test_size", len(X_test))

# # Finalizar a run do MLflow
# mlflow.end_run()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestClassifier

# mlflow.set_tracking_uri("http://localhost:5000")

# # Iniciar uma nova run do MLflow com o nome "PreparacaoDados"
# mlflow.start_run(run_name="PreparacaoDados")

# # Carregar os dados de treinamento e produção
# data_dev = pd.read_parquet("data/dataset_kobe_dev.parquet")
# data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

# # Selecionar as colunas relevantes
# relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
# data_dev_filtered = data_dev[relevant_columns].dropna()
# data_prod_filtered = data_prod[relevant_columns[:-1]].dropna()  # Excluir a coluna 'shot_made_flag' dos dados de produção

# # Salvar o dataset processado
# data_dev_filtered.to_parquet("data/processed/data_filtered.parquet")

# # Separar os dados em treino e teste (80% treino, 20% teste)
# X = data_dev_filtered.drop(columns=['shot_made_flag'])
# y = data_dev_filtered['shot_made_flag']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # Incluir a coluna 'shot_made_flag' nos datasets de treino e teste
# train_data = pd.concat([X_train, y_train], axis=1)
# test_data = pd.concat([X_test, y_test], axis=1)

# # Salvar os datasets de treino e teste em arquivos Parquet
# train_data.to_parquet("data/processed/base_train.parquet")
# test_data.to_parquet("data/processed/base_test.parquet")

# # Registrar os parâmetros e métricas no MLflow
# mlflow.log_param("percent_test", 0.2)
# mlflow.log_metric("train_size", len(X_train))
# mlflow.log_metric("test_size", len(X_test))

# # Registrar uma explicação sobre a escolha de treino e teste
# explanation = """
# A escolha de treino e teste é essencial para avaliar o desempenho do modelo de machine learning. 
# Usamos uma escolha aleatória e estratificada para garantir que as proporções de classes sejam mantidas nos conjuntos de treino e teste. 
# Isso ajuda a evitar viés nos dados e a garantir que o modelo seja avaliado de forma justa.
# Para minimizar os efeitos de viés de dados, podemos aplicar técnicas como validação cruzada, uso de dados de validação, balanceamento de classes e coleta de mais dados quando possível.
# """
# mlflow.log_param("explanation", explanation)

# # Treinar seu modelo (substitua RandomForestClassifier pelo modelo que você está usando)
# sk_model = RandomForestClassifier()
# sk_model.fit(X_train, y_train)

# # Registrar o modelo no MLflow
# mlflow.sklearn.log_model(
#     sk_model, 
#     "PreparacaoDados",
#     registered_model_name="PreparacaoDados"
#     )

# # Finalizar a run do MLflow
# mlflow.end_run()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestClassifier
# import tempfile
# import os

# mlflow.set_tracking_uri("http://localhost:5000")

# # Iniciar uma nova run do MLflow com o nome "PreparacaoDados"
# with mlflow.start_run(run_name="PreparacaoDados"):

#     # Carregar os dados de treinamento e produção
#     data_dev = pd.read_parquet("data/dataset_kobe_dev.parquet")
#     data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

#     # Selecionar as colunas relevantes
#     relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
#     data_dev_filtered = data_dev[relevant_columns].dropna()
#     data_prod_filtered = data_prod[relevant_columns[:-1]].dropna()  # Excluir a coluna 'shot_made_flag' dos dados de produção

#     # Salvar o dataset processado
#     data_dev_filtered.to_parquet("data/processed/data_filtered.parquet")

#     # Separar os dados em treino e teste (80% treino, 20% teste)
#     X = data_dev_filtered.drop(columns=['shot_made_flag'])
#     y = data_dev_filtered['shot_made_flag']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     # Incluir a coluna 'shot_made_flag' nos datasets de treino e teste
#     train_data = pd.concat([X_train, y_train], axis=1)
#     test_data = pd.concat([X_test, y_test], axis=1)

#     # Salvar os datasets de treino e teste em arquivos Parquet
#     train_data.to_parquet("data/processed/base_train.parquet")
#     test_data.to_parquet("data/processed/base_test.parquet")

#     # Calcular o percentual de teste
#     percent_test = len(test_data) / (len(train_data) + len(test_data))

#     # Registrar os parâmetros e métricas no MLflow
#     mlflow.log_param("percent_test", percent_test)
#     mlflow.log_metric("train_size", len(train_data))
#     mlflow.log_metric("test_size", len(test_data))

#     # Registrar uma explicação sobre a escolha de treino e teste
#     explanation = """
#     A escolha de treino e teste é essencial para avaliar o desempenho do modelo de machine learning. 
#     Usamos uma escolha aleatória e estratificada para garantir que as proporções de classes sejam mantidas nos conjuntos de treino e teste. 
#     Isso ajuda a evitar viés nos dados e a garantir que o modelo seja avaliado de forma justa.
#     Para minimizar os efeitos de viés de dados, podemos aplicar técnicas como validação cruzada, uso de dados de validação, balanceamento de classes e coleta de mais dados quando possível.
#     """
    
#     # Criar um arquivo temporário para salvar a explicação
#     with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
#         tmp_file.write(explanation)

#     # Registrar o arquivo temporário como um artefato no MLflow
#     mlflow.log_artifact(tmp_file.name, "explanation.txt")

#     # Remover o arquivo temporário
#     os.unlink(tmp_file.name)

#     # Treinar seu modelo (substitua RandomForestClassifier pelo modelo que você está usando)
#     sk_model = RandomForestClassifier()
#     sk_model.fit(X_train, y_train)

#     # Registrar o modelo no MLflow
#     mlflow.sklearn.log_model(
#         sk_model, 
#         "PreparacaoDados",
#         registered_model_name="PreparacaoDados"
#     )

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os

mlflow.set_tracking_uri("http://localhost:5000")

# Iniciar uma nova run do MLflow com o nome "PreparacaoDados"
with mlflow.start_run(run_name="PreparacaoDados"):

    # Carregar os dados de treinamento e produção
    data_dev = pd.read_parquet("data/dataset_kobe_dev.parquet")
    data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

    # Selecionar as colunas relevantes
    relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    data_dev_filtered = data_dev[relevant_columns].dropna()
    data_prod_filtered = data_prod[relevant_columns[:-1]].dropna()  # Excluir a coluna 'shot_made_flag' dos dados de produção

    # Salvar o dataset processado
    data_dev_filtered.to_parquet("data/processed/data_filtered.parquet")

    # Separar os dados em treino e teste (80% treino, 20% teste)
    X = data_dev_filtered.drop(columns=['shot_made_flag'])
    y = data_dev_filtered['shot_made_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Incluir a coluna 'shot_made_flag' nos datasets de treino e teste
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Salvar os datasets de treino e teste em arquivos Parquet
    train_data.to_parquet("data/processed/base_train.parquet")
    test_data.to_parquet("data/processed/base_test.parquet")

    # Calcular o percentual de teste
    percent_test = len(test_data) / (len(train_data) + len(test_data))

    # Registrar os parâmetros e métricas no MLflow
    mlflow.log_param("percent_test", percent_test)
    mlflow.log_metric("train_size", len(train_data))
    mlflow.log_metric("test_size", len(test_data))

    # Registrar uma explicação sobre a escolha de treino e teste
    explanation = """
    A escolha de treino e teste é essencial para avaliar o desempenho do modelo de machine learning. 
    Usamos uma escolha aleatória e estratificada para garantir que as proporções de classes sejam mantidas nos conjuntos de treino e teste. 
    Isso ajuda a evitar viés nos dados e a garantir que o modelo seja avaliado de forma justa.
    Para minimizar os efeitos de viés de dados, podemos aplicar técnicas como validação cruzada, uso de dados de validação, balanceamento de classes e coleta de mais dados quando possível.
    """
    
    # Criar um arquivo temporário para salvar a explicação
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write(explanation)

    # Registrar o arquivo temporário como um artefato no MLflow
    mlflow.log_artifact(tmp_file.name, "explanation.txt")

    # Remover o arquivo temporário
    os.unlink(tmp_file.name)

    # Treinar seu modelo (substitua RandomForestClassifier pelo modelo que você está usando)
    sk_model = RandomForestClassifier()
    sk_model.fit(X_train, y_train)

    # Registrar o modelo no MLflow
    mlflow.sklearn.log_model(
        sk_model, 
        "PreparacaoDados",
        registered_model_name="PreparacaoDados",
        
    )

    # Plot histograms for numeric variables
    for column in ['lat', 'lon', 'minutes_remaining', 'period', 'shot_distance']:
        plt.figure()
        sns.histplot(train_data[column], kde=True)
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f"output/histogram_{column}.png")
        plt.close()
        mlflow.log_artifact(f"output/histogram_{column}.png")

    # Plot bar plots for categorical variables
    for column in ['playoffs', 'shot_made_flag']:
        plt.figure()
        sns.countplot(x=column, data=train_data)
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(f"output/countplot_{column}.png")
        plt.close()
        mlflow.log_artifact(f"output/countplot_{column}.png")
