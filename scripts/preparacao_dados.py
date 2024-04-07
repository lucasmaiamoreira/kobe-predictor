import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Iniciar uma nova run do MLflow com o nome "PreparacaoDados"
mlflow.start_run(run_name="PreparacaoDados")

# Carregar os dados de treinamento e produção
data_dev = pd.read_parquet("data/dataset_kobe_dev.parquet")
data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

# Selecionar as colunas relevantes
relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
data_dev_filtered = data_dev[relevant_columns].dropna()
data_prod_filtered = data_prod[relevant_columns[:-1]].dropna()  # Excluir a coluna 'shot_made_flag' dos dados de produção

# Salvar o dataset processado
data_dev_filtered.to_parquet("data/processed/data_filtered_dev.parquet")
data_prod_filtered.to_parquet("data/processed/data_filtered_prod.parquet")

# Separar os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(data_dev_filtered.drop(columns=['shot_made_flag']), 
                                                    data_dev_filtered['shot_made_flag'], 
                                                    test_size=0.2, 
                                                    stratify=data_dev_filtered['shot_made_flag'], 
                                                    random_state=42)

# Incluir a coluna 'shot_made_flag' nos datasets de treino e teste
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Salvar os datasets de treino e teste em arquivos Parquet
train_data.to_parquet("data/processed/base_train.parquet")
test_data.to_parquet("data/processed/base_test.parquet")

# Registrar os parâmetros e métricas no MLflow
mlflow.log_param("percent_test", 0.2)
mlflow.log_metric("train_size", len(X_train))
mlflow.log_metric("test_size", len(X_test))

# Finalizar a run do MLflow
mlflow.end_run()
